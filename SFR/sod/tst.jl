# The program that sent to Jonas and Julian

using KitBase, FluxReconstruction, OrdinaryDiffEq, Langevin, LinearAlgebra, Plots
using ProgressMeter: @showprogress

begin
    x0 = 0
    x1 = 1
    ncell = 100
    nface = ncell + 1
    dx = (x1 - x0) / ncell
    deg = 3 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.05
    dt = cfl * dx
    t = 0.0

    uqMethod = "galerkin"
    nr = 9
    nRec = 18
    opType = "uniform"
    parameter1 = 0.95
    parameter2 = 1.05
end
ps = FRPSpace1D(x0, x1, ncell, deg)
uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)

cd(@__DIR__)
include("rhs.jl")
include("filter.jl")

tspan = (0.0, 0.15)
p = (ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, γ, uq)

begin
    V = vandermonde_matrix(ps.deg,ps.xpl)
    VInv = inv(V)
end

u = zeros(ncell, nsp, 3, uq.nm+1)
# stochastic density
#=for i = 1:ncell, j = 1:nsp
    prim = zeros(3, uq.nm+1)
    if ps.x[i] <= 0.5
        prim[1, :] .= uq.pce
        prim[2, 1] = 0.0
        prim[3, 1] = 0.5
    else
        prim[:, 1] .= [0.125, 0.0, 0.625]
    end

    u[i, j, :, :] .= uq_prim_conserve(prim, γ, uq)
end=#

# stochastic location
for i = 1:ncell, j = 1:nsp
    prim = zeros(3, uq.nq)
    for k = 1:uq.nq
        if ps.x[i] <= 0.5 + 0.05 * uq.op.quad.nodes[k]
            prim[:, k] .= [1.0, 0.0, 0.5]
        else
            prim[:, k] .= [0.4, 0.0, 0.625]
        end
    end
    prim_chaos = zeros(3, uq.nm+1)
    for k = 1:3
        prim_chaos[k, :] .= ran_chaos(prim[k, :], uq)
    end
    u[i, j, :, :] .= uq_prim_conserve(prim_chaos, γ, uq)
end

# pre-filtering
for j = 1:size(u, 1)
    for s = 1:size(u, 3)
        uModal = VInv * u[j, :, s, :]
        #FR.modal_filter!(uModal, 15e-2, 10e-5; filter = :l2opt)
        FR.modal_filter!(uModal, 5e-2, 1e-4; filter = :l2)
        u[j, :, s, :] .= V * uModal
    end
end

prob = ODEProblem(dudt!, u, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)

    # filter
    for j = 1:size(itg.u,1)
        for s = 1:size(itg.u,3)
            uModal = VInv * itg.u[j, :, s, :]
            FR.modal_filter!(uModal, 10, 10; filter = :exp)
            #FR.modal_filter!(uModal, 0.8e-2, 1e-5; filter = :l2)
            #FR.modal_filter!(uModal, 5e-2, 1e-5; filter = :l2opt)
            itg.u[j, :, s, :] .= V * uModal
        end
    end
end

sol = zeros(ncell, nsp, 3, 2)
for i in axes(sol, 1), j in axes(sol, 2)
    p1 = zeros(3, uq.nm+1)
    p1 = uq_conserve_prim(itg.u[i, j, :, :], γ, uq)
    p1[end, :] .= lambda_tchaos(p1[end, :], 1.0, uq)

    for k = 1:3
        sol[i, j, k, 1] = mean(p1[k, :], uq.op)
        sol[i, j, k, 2] = std(p1[k, :], uq.op)
    end
end

pic1 = plot(ps.x, sol[:, 2, 1, 1], label="mean", xlabel="x", ylabel="ρ")
pic2 = plot(ps.x, sol[:, 2, 1, 2], label="std")
plot(pic1, pic2)

#sol0 = deepcopy(sol)
plot(ps.x, sol[:, 2, 1, 1], label="Optimized L2", xlabel="x", ylabel="ρ")
plot!(ps.x, sol0[:, 2, 1, 1], label="L2")

plot(ps.x, sol[:, 2, 1, 2], label="Optimized L2", xlabel="x", ylabel="ρ")
plot!(ps.x, sol0[:, 2, 1, 2], label="L2")