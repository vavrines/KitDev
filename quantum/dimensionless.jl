using CairoMakie
import KitBase as KB

cd(@__DIR__)
include("tools.jl")

β = 2
vs = KB.VSpace1D(-5, 5, 100)
prim = [0.3, 0.3, 0.8]

f = fd_equilibrium(vs.u, prim, β)

begin
    fig = lines(vs.u, f; label = "f")
    axislegend()
    fig
end

w1 = KB.moments_conserve(f, vs.u, vs.weights)

function fd_moments(prim, β)
    w = zeros(3)
    w[1] = fd_integral(-0.5, prim[1]) * β / sqrt(prim[3])
    w[2] = w[1] * prim[2]
    w[3] = fd_integral(0.5, prim[1]) * β / (prim[3])^(3/2) / 4 + 0.5 * w[1] * prim[2]^2

    return w
end

w2 = fd_moments(prim, β)
