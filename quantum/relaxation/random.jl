using OrdinaryDiffEq, CairoMakie, Langevin
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

cd(@__DIR__)
include("../tools.jl")

tspan = (0, 10)
tsteps = tspan[1]:0.01:tspan[2]
tnum = length(tsteps)

vs = VSpace1D(-6.0, 6.0, 200)
unum = vs.nu

β = 2
f0 = vs.u .^ 2 .* exp.(-vs.u .^ 2)
w0 = moments_conserve(f0, vs.u, vs.weights)
prim0 = quantum_conserve_prim(w0, β)
F0 = fd_equilibrium(vs.u, prim0, β)
#τ0 = gas.Kn / w0[1]

# nr, nrec, μ, σ, op, method
uq = UQ1D(5, 40, 1.0, 0.2, "gauss", "galerkin")
ν = uq.pce

finit = zeros(uq.nr + 1, unum)
finit[1, :] .= f0

function ODEGalerkinTen(du, u, p, t)
    L = size(u, 1) - 1
    for m = 0:L
        for i = 1:unum
            du[m+1, i] = (
                F0[i] * p[m+1] - sum(
                    p[j+1] * u[k+1, i] * uq.t3Product[j, k, m] / uq.t2Product[m, m]
                    for j = 0:L for k = 0:L
                )
            )
        end
    end
end

probGalerkinTen = ODEProblem(ODEGalerkinTen, finit, (tspan[1], tspan[2]), ν)
solGalerkinTen = solve(
    probGalerkinTen,
    Tsit5(),
    abstol = 1e-10,
    reltol = 1e-10,
    saveat = tsteps,
)

sol = solGalerkinTen |> Array

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f",
    title = "")
    lines!(tsteps, sol[1, 100, :]; label = "quantum")
    axislegend()
    fig
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "t",
    title = "")
    contourf!(sol[1, :, :])
    fig
end
