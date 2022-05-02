using NonlinearSolve, CairoMakie
import KitBase as KB
using KitBase.OffsetArrays, KitBase.StructArrays
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)
include("tools.jl")

set = KB.Setup(
    space = "1d1f1v",
    interpOrder = 1,
    boundary = ["fix", "fix"],
    maxTime = 0.18,
    cfl = 0.5,
)

ps = KB.PSpace1D(0, 1, 100, 1)
vs = KB.VSpace1D(-5, 5, 60)
gas = KB.Gas(Kn = 1e-4, γ = 2)

function ib_condition(set, ps, vs, gas)
    primL = [0.8, 0.0, 1.0]
    primR = [0.7, 0.0, 1.25]
    β = gas.γ

    wL = prim_conserve(primL, β)
    wR = prim_conserve(primR, β)

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
        prim = conserve_prim(w, p.β)
        h = fd_equilibrium(p.u, prim, p.β)
        return h
    end

    return fw, ff, bc, p
end

fw, ff, bc, p = ib_condition(set, ps, vs, gas)
ib = KB.IB1F{typeof(bc)}(fw, ff, bc, p)

ks = KB.SolverSet(set, ps, vs, gas, ib, @__DIR__)

ctr = OffsetArray{KB.ControlVolume1F}(undef, axes(ks.ps.x, 1))
for i in axes(ctr, 1)
    w = ks.ib.fw(ks.ps.x[i], ks.ib.p)
    prim = conserve_prim(w, ks.gas.γ)
    h = fd_equilibrium(vs.u, prim, ks.gas.γ)
    ctr[i] = KB.ControlVolume(w, prim, h, 1)
end
ctr = StructArray(ctr)

face = Array{KB.Interface1F}(undef, ks.ps.nx + 1)
for i = 1:ks.ps.nx+1
    face[i] = KB.Interface(
        zero(ks.ib.fw(ks.ps.x[1], ks.ib.p)),
        zero(ks.ib.ff(ks.ps.x[1], ks.ib.p)),
        1,
    )
end

res = zeros(3)
dt = KB.timestep(ks, ctr, 0.0)
nt = floor(ks.set.maxTime / dt) |> Int

function step!(
    fwL::T1,
    ffL::T2,
    w::T3,
    prim::T3,
    f::T4,
    fwR::T1,
    ffR::T2,
    u::T5,
    β,
    Kn,
    dx,
    dt,
    RES,
    AVG,
) where {T1,T2,T3,T4,T5}

    #--- store W^n and calculate H^n,\tau^n ---#
    w_old = deepcopy(w)

    #--- update W^{n+1} ---#
    @. w += (fwL - fwR) / dx
    prim .= conserve_prim(w, β)

    #--- record residuals ---#
    @. RES += (w - w_old)^2
    @. AVG += abs(w)

    #--- calculate M^{n+1} and tau^{n+1} ---#
    M = fd_equilibrium(u, prim, β)
    τ = Kn / w[1]

    #--- update distribution function ---#
    for i in eachindex(u)
        f[i] = (f[i] + (ffL[i] - ffR[i]) / dx + dt / τ * M[i]) / (1.0 + dt / τ)
    end

end

function update!(
    KS::KB.AbstractSolverSet,
    ctr::AbstractVector{TC},
    face::AbstractVector{TF},
    dt,
    residual
) where {TC,TF}

    sumRes = zero(ctr[1].w)
    sumAvg = zero(ctr[1].w)

    @inbounds @threads for i = 1:KS.pSpace.nx
        step!(
            face[i].fw,
            face[i].ff,
            ctr[i].w,
            ctr[i].prim,
            ctr[i].f,
            face[i+1].fw,
            face[i+1].ff,
            KS.vSpace.u,
            KS.gas.γ,
            KS.gas.Kn,
            KS.ps.dx[i],
            dt,
            sumRes,
            sumAvg,
        )
    end
    
    for i in eachindex(residual)
        residual[i] = sqrt(sumRes[i] * KS.pSpace.nx) / (sumAvg[i] + 1.e-7)
    end

    return nothing
end

@showprogress for iter = 1:nt
    KB.evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)
end

sol = zeros(ks.ps.nx, 6)
for i = 1:ks.ps.nx
    sol[i, 1:3] .= ctr[i].w
    sol[i, 4:6] .= conserve_prim(ctr[i].w, ks.gas.γ)
end
begin
    fig = lines(ks.ps.x[1:ks.ps.nx], 1 ./ sol[:, 6]; label = "f", xlabel="x")
    axislegend()
    fig
end
