using CairoMakie
import KitBase as KB

cd(@__DIR__)
include("integral.jl")

function fd_equilibrium(u, prim, β)
    A, U, λ = prim
    return @. β / sqrt(π) * (A^(-1) * exp(λ * (u - U)^2) + 1)^(-1)
end

β = 2
vs = KB.VSpace1D(-5, 5, 100)
prim = [0.3, 0.0, 1.0]

f = fd_equilibrium(vs.u, prim, β)

begin
    fig = lines(vs.u, f; label = "f")
    axislegend()
    fig
end

w1 = KB.moments_conserve(f, vs.u, vs.weights)
w1[1]

fd_integral(-0.5, prim[1]) * β * sqrt(prim[3])

fd_integral(0.5, prim[1]) * β * (prim[3])^(2/3) / 4
