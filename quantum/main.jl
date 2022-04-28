using CairoMakie
import KitBase as KB

cd(@__DIR__)
include("integral.jl")

function equilibrium(u, prim, β)
    A, U, λ = prim
    return @. β / sqrt(π) * (A^(-1) * exp(λ * (u - U)^2) - 1)^(-1)
end

β = 1
vs = KB.VSpace1D(-8, 8, 100)
prim = [0.3, 0.2, 1.2]

f = equilibrium(vs.u, prim, β)

begin
    fig = lines(vs.u, f; label = "f")
    axislegend()
    fig
end

w1 = KB.moments_conserve(f, vs.u, vs.weights)

w1[1]
fd_integral(-0.5, prim[1])

sqrt(prim[3]) * fd_integral(0.5, prim[1])

w1[2] / w1[1]
