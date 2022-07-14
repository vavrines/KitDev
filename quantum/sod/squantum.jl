using CairoMakie, Langevin
using KitBase.OffsetArrays, KitBase.StructArrays, KitBase.JLD2, KitBase.SpecialFunctions
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)
include("../tools.jl")

set = KB.Setup(
    space = "1d1f1v",
    interpOrder = 2,
    boundary = ["fix", "fix"],
    maxTime = 0.12,
    cfl = 0.5,
)

ps = KB.PSpace1D(0, 1, 100, 1)
vs = KB.VSpace1D(-5, 5, 60)
gas = KB.Gas(Kn = 1e-3, γ = 2)

function ib_condition(set, ps, vs, gas)
    primL = [0.5, 0.0, 0.5] # [A, U, λ]
    primR = [0.1, 0.0, 0.625]
    β = gas.γ

    wL = quantum_prim_conserve(primL, β)
    wR = quantum_prim_conserve(primR, β)

    p = (
        x0 = ps.x0,
        x1 = ps.x1,
        wL = wL,
        wR = wR,
        primL = primL,
        primR = primR,
        β = gas.γ,
    )

    fw = function (x, p)
        if x <= (p.x0 + p.x1) / 2
            return p.wL
        else
            return p.wR
        end
    end

    bc = function (x, p)
        if x <= (p.x0 + p.x1) / 2
            return p.primL
        else
            return p.primR
        end
    end

    p = (p..., u = vs.u)
    ff = function (x, p)
        w = ifelse(x <= (p.x0 + p.x1) / 2, p.wL, p.wR)
        prim = quantum_conserve_prim(w, p.β)
        h = fd_equilibrium(p.u, prim, p.β)
        return h
    end

    return fw, ff, bc, p
end

fw, ff, bc, p = ib_condition(set, ps, vs, gas)
ib = KB.IB1F{typeof(bc)}(fw, ff, bc, p)

ks = KB.SolverSet(set, ps, vs, gas, ib, @__DIR__)

uq = UQ1D(5, 10, 0.8, 1.2, "uniform", "galerkin")

wL0 = ks.ib.fw(ks.ps.x[1], ks.ib.p)
wLq = zeros(3, uq.nq)
for i = 1:uq.nq
    wLq[:, i] .= wL0 .* uq.pceSample[i]
end
wR0 = ks.ib.fw(ks.ps.x[end], ks.ib.p)
wRq = zeros(3, uq.nq)
for i = 1:uq.nq
    wRq[:, i] .= wR0 .* uq.pceSample[i]
end

primLq = zeros(3, uq.nq)
primRq = zeros(3, uq.nq)
for j = 1:uq.nq
    primLq[:, j] .= quantum_conserve_prim(wLq[:, j], gas.γ)
    primRq[:, j] .= quantum_conserve_prim(wRq[:, j], gas.γ)
end

fLq = zeros(ks.vs.nu, uq.nq)
for j = 1:uq.nq
    fLq[:, j] .= fd_equilibrium(vs.u, primLq[:, j], ks.gas.γ)
end
fRq = zeros(ks.vs.nu, uq.nq)
for j = 1:uq.nq
    fRq[:, j] .= fd_equilibrium(vs.u, primRq[:, j], ks.gas.γ)
end

wL = zeros(3, uq.nr+1)
wR = zeros(3, uq.nr+1)
primL = zeros(3, uq.nr+1)
primR = zeros(3, uq.nr+1)
fL = zeros(ks.vs.nu, uq.nr+1)
fR = zeros(ks.vs.nu, uq.nr+1)
for i = 1:3
    wL[i, :] .= ran_chaos(wLq[i, :], uq)
    wR[i, :] .= ran_chaos(wRq[i, :], uq)
    primL[i, :] .= ran_chaos(primLq[i, :], uq)
    primR[i, :] .= ran_chaos(primRq[i, :], uq)
end
for i = 1:ks.vs.nu
    fL[i, :] .= ran_chaos(fLq[i, :], uq)
    fR[i, :] .= ran_chaos(fRq[i, :], uq)
end

ctr = OffsetArray{KB.ControlVolume1F}(undef, axes(ks.ps.x, 1))
for i in axes(ctr, 1)
    if i <= ks.ps.nx÷2
        ctr[i] = KB.ControlVolume(wL, primL, fL, 1)
    else
        ctr[i] = KB.ControlVolume(wR, primR, fR, 1)
    end
end
#ctr = StructArray(ctr)

face = Array{KB.Interface1F}(undef, ks.ps.nx + 1)
for i = 1:ks.ps.nx+1
    face[i] = KB.Interface(
        zeros(3, uq.nr+1),
        zeros(ks.vs.nu, uq.nr+1),
        1,
    )
end

res = zeros(3)
dt = Δt(ks, ctr, 0.0, uq)
nt = floor(ks.set.maxTime / dt) |> Int

function Δt(KS, ctr, simTime, uq)
    tmax = 0.0
    Threads.@threads for i = 1:KS.pSpace.nx
        @inbounds prim = ctr[i].prim
        sos = uq_sound_speed(prim, KS.gas.γ, uq)
        vmax = max(maximum(KS.vSpace.u1), maximum(abs.(prim[2, :]))) + maximum(sos)
        tmax = max(tmax, vmax / KS.ps.dx[i])
    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt
end

function st!(KS, uq, faceL, cell, faceR,
    dt,
    dx,
    RES,
    AVG,
    coll = :bgk::Symbol,
) where {T1<:AbstractSolverSet} # 1D1F1V
    #--- update conservative flow variables: step 1 ---#
    # w^n
    w_old = deepcopy(cell.w)
    prim_old = deepcopy(cell.prim)

    # flux -> w^{n+1}
    @. cell.w += (faceL.fw - faceR.fw) / dx

    wRan = chaos_ran(cell.w, 2, uq)

    primRan = zero(wRan)
    for j in axes(wRan, 2)
        primRan[:, j] .= quantum_conserve_prim(wRan[:, j], KS.gas.γ)
    end

    # locate variables on random quadrature points

    #primRan = chaos_ran(cell.prim, 2, uq)

    #cell.w .= ran_chaos(wRan, 2, uq)
    cell.prim .= ran_chaos(primRan, 2, uq)

    #--- update particle distribution function ---#
    # flux -> f^{n+1}
    #@. cell.f += (faceL.ff - faceR.ff) / cell.dx

    fRan =
        chaos_ran(cell.f, 2, uq) .+
        (chaos_ran(faceL.ff, 2, uq) .- chaos_ran(faceR.ff, 2, uq)) ./ dx

    # source -> f^{n+1}
    tau = [KS.gas.Kn / wRan[1, j] for j in axes(wRan, 2)]

    gRan = zeros(KS.vSpace.nu, uq.op.quad.Nquad)
    for j in axes(gRan, 2)
        gRan[:, j] .= fd_equilibrium(KS.vs.u, primRan[:, j], ks.gas.γ)
    end

    # BGK term
    for j in axes(fRan, 2)
        @. fRan[:, j] = (fRan[:, j] + dt / tau[j] * gRan[:, j]) / (1.0 + dt / tau[j])
    end

    cell.f .= ran_chaos(fRan, 2, uq)

    #--- record residuals ---#
    @. RES += (w_old[:, 1] - cell.w[:, 1])^2
    @. AVG += abs(cell.w[:, 1])
end

function up!(KS, uq, ctr, face, dt, residual; coll = :bgk)
    sumRes = zeros(3)
    sumAvg = zeros(3)

    @inbounds Threads.@threads for i = 2:KS.pSpace.nx-1
        st!(KS, uq, face[i], ctr[i], face[i+1], dt, KS.ps.dx[i], sumRes, sumAvg, coll)
    end

    for i in axes(residual, 1)
        residual[i] = sqrt(sumRes[i] * KS.pSpace.nx) / (sumAvg[i] + 1.e-7)
    end
end

function ev!(KS, uq, ctr, face, dt)
    @inbounds Threads.@threads for i in eachindex(face)
        ufgalerkin!(
            KS,
            uq,
            ctr[i-1],
            face[i],
            ctr[i],
            dt,
            KS.ps.dx[i-1],
            KS.ps.dx[i],
        )
    end
end

function ufgalerkin!(KS, uq, cellL, face, cellR, dt, dxL, dxR)
    @inbounds for j in axes(cellL.f, 2)
        fw = @view face.fw[:, j]
        ff = @view face.ff[:, j]

        flux_kfvs!(
            fw,
            ff,
            cellL.f[:, j] .+ 0.5 .* dxL .* cellL.sf[:, j],
            cellR.f[:, j] .- 0.5 .* dxR .* cellR.sf[:, j],
            KS.vSpace.u,
            KS.vSpace.weights,
            dt,
            cellL.sf[:, j],
            cellR.sf[:, j],
        )
    end
end

@showprogress for iter = 1:nt
    ev!(ks, uq, ctr, face, dt)
    up!(ks, uq, ctr, face, dt, res)
end

sol = zeros(ks.ps.nx, 3)
for i in axes(sol, 1)
    sol[i, :] .= ctr[i].w[:, 1]
end

begin
    fig = lines(ks.ps.x[1:ks.ps.nx], sol[:, 1]; label = "quantum")
    axislegend()
    fig
end
