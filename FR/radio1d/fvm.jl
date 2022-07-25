using KitBase, KitBase.FastGaussQuadrature
using Plots
using ProgressMeter: @showprogress

set = Setup(
    matter = "radiation",
    case = "inflow",
    space = "1d1f1v",
    boundary = "maxwell",
    cfl = 0.2,
    maxTime = 0.5,
)

ps = PSpace1D(0, 1, 100)
vs = VSpace1D(-1, 1, 28)
gas = Radiation(Kn = 1.0)
ib = IB1F(zeros, (x...) -> ones(Float64, vs.nu) .* 1e-3, 1e-3)
ks = SolverSet(set, ps, vs, gas, ib)

f0 = ks.ib.ff()
ctr = Array{ControlVolume1D1F}(undef, ps.nx)
face = Array{Interface1D1F}(undef, ps.nx + 1)
for i in eachindex(ctr)
    ctr[i] = ControlVolume1D1F([sum(vs.weights .* f0)], [sum(vs.weights .* f0)], ks.ib.ff())
end
for i = 1:ks.pSpace.nx+1
    face[i] = Interface1D1F(sum(vs.weights .* f0), ks.ib.ff())
end

function fb!(ff, f, u, dt, rot = 1)
    δ = heaviside.(u .* rot)
    fWall = 0.5 .* δ .+ f .* (1.0 .- δ)
    @. ff = u * fWall * dt

    return nothing
end

function step(
    ffL::T2,
    w::T3,
    f::T4,
    ffR::T2,
    u::T5,
    weights::T5,
    τ,
    dx,
    dt,
) where {
    T2<:AbstractArray{<:AbstractFloat,1},
    T3,
    T4<:AbstractArray{<:AbstractFloat,1},
    T5<:AbstractArray{<:AbstractFloat,1},
}
    M = sum(weights .* f)
    for i in eachindex(u)
        f[i] += (ffL[i] - ffR[i]) / dx + (M - f[i]) / τ * dt
    end
    w[1] = sum(weights .* f)
end

dt = set.cfl * ps.dx[1]
nt = set.maxTime / dt |> floor |> Int

@showprogress for iter = 1:150#nt
    fb!(face[1].ff, ctr[1].f, vs.u, dt)
    @inbounds for i = 2:ps.nx
        flux_kfvs!(face[i].ff, ctr[i-1].f, ctr[i].f, vs.u, dt, ctr[i-1].sf, ctr[i].sf)
    end

    @inbounds for i = 1:ps.nx-1
        step(
            face[i].ff,
            ctr[i].w,
            ctr[i].f,
            face[i+1].ff,
            vs.u,
            vs.weights,
            1.0,
            ps.dx[i],
            dt,
        )
    end
end

begin
    pltx = ks.pSpace.x[1:ks.pSpace.nx]
    plty = zeros(ks.pSpace.nx, 6)
    for i in eachindex(pltx)
        plty[i, 1] = ctr[i].w[1]
    end
    plot(pltx, plty[:, 1], label = "Density", lw = 2, xlabel = "x")
end
