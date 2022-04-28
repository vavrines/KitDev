using CairoMakie
import KitBase as KB

cd(@__DIR__)
include("integral.jl")

h = 6.62607015e-34
m = 9.1093837e-31
β = 2
θ = h / m / β
k = 1.380649e-23
ℓ = h / sqrt(2 * π * m * k * 273)

function equilibrium(u, prim)
    A, U, λ = prim
    return @. m * θ^(-1) * (A^(-1) * exp(λ * (u - U)^2) + 1)^(-1)
end

T = 273
λ = m / 2 / k / T
prim = [0.3, 0, λ]

vs = KB.VSpace1D(-4 * sqrt(1 / λ), 4 * sqrt(1 / λ), 100)

f = equilibrium(vs.u, prim)

begin
    fig = lines(vs.u, f; label = "f")
    axislegend()
    fig
end

w1 = KB.moments_conserve(f, vs.u, vs.weights)
w1[1]

fd_integral(-0.5, prim[1]) * m / ℓ * β
