using NonlinearSolve, CairoMakie
using KitBase
using KitBase.OffsetArrays, KitBase.StructArrays, KitBase.JLD2, KitBase.SpecialFunctions
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)
include("../../tools.jl")

set = Setup(
    space = "1d1f1v",
    interpOrder = 2,
    boundary = ["fix", "fix"],
    maxTime = 0.12,
    cfl = 0.5,
)

ps = PSpace1D(0, 1, 100, 1)
vs = VSpace1D(-5, 5, 60)
gas = Gas(Kn = 1e-3, γ = 2) # γ is β in quantum model

function ib_condition(set, ps, vs, gas)
    primL = [20.0, 0.0, 0.5] # [A, U, λ]
    primR = [5.0, 0.0, 0.625]
    β = gas.γ

    wL = quantum_prim_conserve(primL, β)
    wR = quantum_prim_conserve(primR, β)

    p = (x0 = ps.x0, x1 = ps.x1, wL = wL, wR = wR, primL = primL, primR = primR, β = gas.γ)

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
ib = IB1F{typeof(bc)}(fw, ff, bc, p)

ks = SolverSet(set, ps, vs, gas, ib, @__DIR__)
ctr, face = init_fvm(ks)

for i in axes(ctr, 1)
    w = ks.ib.fw(ks.ps.x[i], ks.ib.p)
    prim = quantum_conserve_prim(w, ks.gas.γ)
    h = fd_equilibrium(vs.u, prim, ks.gas.γ)
    ctr[i] = KB.ControlVolume(w, prim, h, 1)
end
ctr = StructArray(ctr)

res = zeros(3)
dt = timestep(ks, ctr, 0.0)
nt = floor(ks.set.maxTime / dt) |> Int

function qstep!(w, prim, f, fwL, ffL, fwR, ffR, u, β, Kn, dx, dt, RES, AVG)
    w_old = deepcopy(w)

    @. w += (fwL - fwR) / dx
    prim .= quantum_conserve_prim(w, β)

    @. RES += (w - w_old)^2
    @. AVG += abs(w)

    M = fd_equilibrium(u, prim, β)
    τ = Kn / w[1]

    for i in eachindex(u)
        f[i] = (f[i] + (ffL[i] - ffR[i]) / dx + dt / τ * M[i]) / (1.0 + dt / τ)
    end
end

function qstep!(KS, cell, faceL, faceR, p, coll = :bgk; st = step!)
    dt, dx, RES, AVG = p
    qstep!(
        cell.w,
        cell.prim,
        cell.f,
        faceL.fw,
        faceL.ff,
        faceR.fw,
        faceR.ff,
        KS.vs.u,
        KS.gas.γ,
        KS.gas.Kn,
        dx,
        dt,
        RES,
        AVG,
    )
end

@showprogress for iter = 1:nt
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res; fn = qstep!)
end

@load "classical.jld2" sol
sol0 = deepcopy(sol)

sol = zeros(ks.ps.nx, 6)
for i = 1:ks.ps.nx
    sol[i, 1:3] .= ctr[i].w
    sol[i, 4:6] .= quantum_conserve_prim(ctr[i].w, ks.gas.γ)
end
begin
    fig = lines(ks.ps.x[1:ks.ps.nx], sol[:, 1]; label = "quantum")
    lines!(ks.ps.x[1:ks.ps.nx], sol0[:, 1]; label = "classical")
    axislegend()
    fig
end
