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
    maxTime = 0.15,
    cfl = 0.5,
)

ps = KB.PSpace1D(0, 1, 100, 1)
vs = KB.VSpace1D(-5, 5, 60)
gas = KB.Gas(Kn = 1e-3, γ = 2)

function ib_condition(set, ps, vs, gas)
    # [A, U, λ]
    primL = [0.5, 0.0, 0.5]
    primR = [0.1, 0.0, 0.625]
    β = gas.γ # quantum

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
ib = KB.IB1F{typeof(bc)}(fw, ff, bc, p)

ks = KB.SolverSet(set, ps, vs, gas, ib)

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

wL = zeros(3, uq.nr + 1)
wR = zeros(3, uq.nr + 1)
primL = zeros(3, uq.nr + 1)
primR = zeros(3, uq.nr + 1)
fL = zeros(ks.vs.nu, uq.nr + 1)
fR = zeros(ks.vs.nu, uq.nr + 1)
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
    if i <= ks.ps.nx ÷ 2
        ctr[i] = KB.ControlVolume(wL, primL, fL, 1)
    else
        ctr[i] = KB.ControlVolume(wR, primR, fR, 1)
    end
end
ctr = StructArray(ctr)

face = Array{KB.Interface1F}(undef, ks.ps.nx + 1)
for i = 1:ks.ps.nx+1
    face[i] = KB.Interface(zeros(3, uq.nr + 1), zeros(ks.vs.nu, uq.nr + 1), 1)
end

res = zeros(3)
dt = timestep(ks, uq, ctr, 0.0)
nt = floor(ks.set.maxTime / dt) |> Int

@showprogress for iter = 1:nt
    evolve!(ks, uq, ctr, face, dt)
    update!(ks, uq, ctr, face, dt, res; fn = st!)
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
