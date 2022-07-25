using NonlinearSolve, CairoMakie
import KitBase as KB
using KitBase.OffsetArrays, KitBase.StructArrays, KitBase.JLD2
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)

set = KB.Setup(
    space = "1d1f1v",
    interpOrder = 2,
    boundary = ["fix", "fix"],
    maxTime = 0.12,
    cfl = 0.5,
)

ps = KB.PSpace1D(0, 1, 100, 1)
vs = KB.VSpace1D(-5, 5, 60)
gas = KB.Gas(Kn = 1e-3, K = 0, γ = 3)

function ib_condition(set, ps, vs, gas)
    # obtained from quantum results
    ## keep same primitive variables
    #=primL = [1.046375208144766, 0.0, 1.0]
    primR = [0.8541199925148495, 0.0, 1.25]

    wL = KB.prim_conserve(primL, gas.γ)
    wR = KB.prim_conserve(primR, gas.γ)=#

    ## keep same conservative variables
    #wL = [1.046375208144766, 0.0, 0.3197013471583676]
    #wR = [0.8541199925148495, 0.0, 0.20486709817348708]
    wL = [5.238106334434739, 0.0, 6.335168390343023]
    wR = [3.2818507271758053, 0.0, 2.3114593015227394]

    primL = KB.conserve_prim(wL, gas.γ)
    primR = KB.conserve_prim(wR, gas.γ)

    p = (x0 = ps.x0, x1 = ps.x1, wL = wL, wR = wR, primL = primL, primR = primR, γ = gas.γ)

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
        prim = KB.conserve_prim(w, p.γ)
        h = KB.maxwellian(p.u, prim)
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
    prim = KB.conserve_prim(w, ks.gas.γ)
    h = KB.maxwellian(vs.u, prim)
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

@showprogress for iter = 1:nt
    KB.evolve!(ks, ctr, face, dt)
    KB.update!(ks, ctr, face, dt, res)
end

sol = zeros(ks.ps.nx, 6)
for i = 1:ks.ps.nx
    sol[i, 1:3] .= ctr[i].w
    sol[i, 4:6] .= ctr[i].prim
end
begin
    fig = lines(ks.ps.x[1:ks.ps.nx], 1 ./ sol[:, 3]; label = "f", xlabel = "x")
    axislegend()
    fig
end

@save "classical.jld2" sol
