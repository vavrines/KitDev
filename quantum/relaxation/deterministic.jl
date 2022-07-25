using OrdinaryDiffEq, CairoMakie
using KitBase
using KitBase.OffsetArrays, KitBase.StructArrays, KitBase.JLD2, KitBase.SpecialFunctions
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

cd(@__DIR__)
include("../tools.jl")

set = KB.Setup(space = "1d1f1v", maxTime = 3.0, cfl = 0.5)
vs = KB.VSpace1D(-6, 6, 200)
gas = KB.Gas(Kn = 1e0, γ = 2) # γ is β in quantum model
β = gas.γ

tspan = (0.0, set.maxTime)
tsteps = KB.linspace(tspan[1], tspan[2], 31)

# quantum
f0 = 0.5 * (1 / π)^0.5 .* (exp.(-(vs.u .- 0.99) .^ 2) .+ exp.(-(vs.u .+ 0.99) .^ 2))
w0 = KB.moments_conserve(f0, vs.u, vs.weights)
prim0 = quantum_conserve_prim(w0, β)
F0 = fd_equilibrium(vs.u, prim0, β)
τ0 = gas.Kn / w0[1]

# classic
prim_classic = KB.conserve_prim(w0, 3)
M0 = KB.maxwellian(vs.u, prim_classic)

prob = ODEProblem(KB.bgk_ode!, f0, tspan, [F0, τ0])
sol = solve(prob, Midpoint(), saveat = tsteps) |> Array

prob1 = ODEProblem(KB.bgk_ode!, f0, tspan, [M0, τ0])
sol1 = solve(prob1, Midpoint(), saveat = tsteps) |> Array

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, sol[:, 11]; label = "quantum")
    lines!(vs.u, sol1[:, 11]; label = "classical", linestyle = :dash)
    lines!(vs.u, F0; label = "equilibrium")
    axislegend()
    fig
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, sol[:, 21]; label = "quantum")
    lines!(vs.u, sol1[:, 21]; label = "classical", linestyle = :dash)
    lines!(vs.u, F0; label = "equilibrium")
    axislegend()
    fig
end
