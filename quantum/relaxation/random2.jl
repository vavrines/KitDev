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

uq = UQ1D(5, 40, 0.8, 1.2, "uniform", "galerkin")

f0q = zeros(uq.nq, vs.nu)
for i = 1:uq.nq, j = 1:unum
    f0q[i, j] = uq.pceSample[i] * vs.u[j] ^ 2 * exp(-vs.u[j] ^ 2)
end

f0 = zeros(uq.nr + 1, unum)
for j in axes(f0, 2)
    f0[:, j] .= ran_chaos(f0q[:, j], uq)
end

w0q = zeros(uq.nq, 3)
for i in axes(w0q, 1)
    w0q[i, :] .= moments_conserve(f0q[i, :], vs.u, vs.weights)
end

prim0q = zero(w0q)
for i in axes(w0q, 1)
    prim0q[i, :] .= quantum_conserve_prim(w0q[i, :], β)
end

F0q = zeros(uq.nq, unum)
for i = 1:uq.nq, j = 1:unum
    F0q[i, :] .= fd_equilibrium(vs.u, prim0q[i, :], β)
end

w0 = zeros(uq.nr + 1, 3)
prim0 = zero(w0)
F0 = zeros(uq.nr+1, unum)
for j = 1:3
    w0[:, j] .= ran_chaos(w0q[:, j], uq)
    prim0[:, j] .= ran_chaos(prim0q[:, j], uq)
end
for j = 1:unum
    F0[:, j] .= ran_chaos(F0q[:, j], uq)
end

ν = [1.0; zeros(uq.nr)]

function ODEGalerkinTen(du, u, p, t)
    L = size(u, 1) - 1
    for m = 0:L
        for i = 1:unum
            du[m+1, i] = sum(
                    p[j+1] * F0[k+1, i] * uq.t3Product[j, k, m] / uq.t2Product[m, m]
                    for j = 0:L for k = 0:L
                ) -
                sum(
                    p[j+1] * u[k+1, i] * uq.t3Product[j, k, m] / uq.t2Product[m, m]
                    for j = 0:L for k = 0:L
                )
        end
    end
end

probGalerkinTen = ODEProblem(ODEGalerkinTen, f0, (tspan[1], tspan[2]), ν)
solGalerkinTen = solve(
    probGalerkinTen,
    Tsit5(),
    abstol = 1e-10,
    reltol = 1e-10,
    saveat = tsteps,
)

sol = solGalerkinTen |> Array

solMean = zeros(unum, tnum)
solStd = zeros(unum, tnum)
for i in axes(solMean, 1), j in axes(solMean, 2)
    solMean[i, j] = mean(sol[:, i, j], uq.op)
    solStd[i, j] = std(sol[:, i, j], uq.op)
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "t", title = "")
    co = contourf!(solMean)
    Colorbar(fig[1, 2], co)
    fig
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "t", title = "")
    co = contourf!(solStd)
    Colorbar(fig[1, 2], co)
    fig
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f",
    title = "")
    lines!(vs.u, solStd[:, end]; label = "quantum")
    axislegend()
    fig
end
