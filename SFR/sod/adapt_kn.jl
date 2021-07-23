using KitBase, FluxReconstruction, OrdinaryDiffEq, Langevin, LinearAlgebra, Plots
using ProgressMeter: @showprogress

begin
    x0 = 0
    x1 = 1
    ncell = 100
    nface = ncell + 1
    dx = (x1 - x0) / ncell
    deg = 2 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.05
    dt = cfl * dx / (2.0)
    t = 0.0
    tspan = (0.0, 0.15)

    uqMethod = "galerkin"
    nr = 9
    nRec = 18
    opType = "uniform"
    parameter1 = 0.95
    parameter2 = 1.05
end
ps = FRPSpace1D(x0, x1, ncell, deg)
uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)
V = vandermonde_matrix(ps.deg,ps.xpl)
VInv = inv(Array(ps.V))

cd(@__DIR__)
include("rhs.jl")
include("filter.jl")

begin
    isRandomLocation = true
    isPrefilter = false#true

    u = zeros(ncell, nsp, 3, uq.nm+1)
    if isRandomLocation
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
    else
        # stochastic density
        for i = 1:ncell, j = 1:nsp
            prim = zeros(3, uq.nm+1)
            if ps.x[i] <= 0.5
                prim[1, :] .= uq.pce
                #prim[1, 1] = 1.0
                prim[2, 1] = 0.0
                prim[3, 1] = 0.5
            else
                prim[:, 1] .= [0.125, 0.0, 0.625]
            end

            u[i, j, :, :] .= uq_prim_conserve(prim, γ, uq)
        end
    end

    if isPrefilter
        # pre-filtering
        for j = 1:size(u, 1)
            for s = 1:size(u, 3)
                uModal = VInv * u[j, :, s, :]
                #FR.modal_filter!(uModal, 15e-2, 10e-5; filter = :l2opt)
                FR.modal_filter!(uModal, 5e-2, 5e-2; filter = :l2)
                #FR.modal_filter!(uModal, 5, 5; filter = :exp)
                u[j, :, s, :] .= V * uModal
            end
        end
    end
end

p = (ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, γ, uq)
prob = ODEProblem(dudt!, u, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

function detector(Se, deg, S0 = -3.0 * log10(deg), κ = 4.0)
    if Se < S0 - κ
        σ = 1.0
    elseif S0 - κ <= Se < S0 + κ
        σ = 0.5 * (1.0 - sin(0.5 * π * (Se - S0) / κ))
    else
        σ = 0.0
    end

    return σ < 0.99 ? true : false
end

@showprogress for iter = 1:nt
    step!(itg)

    for i = 1:size(itg.u, 1)
        #=
        ũ = VInv * itg.u[i, :, 1, 1]
        su = log10(ũ[end]^2 / sum(ũ.^2))
        isShock = detector(su, ps.deg)
        if isShock
            for j in axes(itg.u, 3), k in axes(itg.u, 4)
                û = VInv * itg.u[i, :, j, k]
                FR.modal_filter!(û, 1e-2; filter = :l2)
                itg.u[i, :, j, k] .= ps.V * û
            end
        end

        ṽ = itg.u[i, end÷2, 1, :]
        sv = log10(ṽ[end]^2 / sum(ṽ.^2))
        isShock = detector(sv, ps.deg)
        if isShock
            for j in axes(itg.u, 2), k in axes(itg.u, 3)
                ṽ = @view itg.u[i, j, k, :]
                FR.modal_filter!(ṽ, 5e-4; filter = :l2)
            end
        end=#

        ũ = VInv * itg.u[i, :, 1, 1]
        ṽ = itg.u[i, end÷2, 1, :]
        su = log10(ũ[end]^2 / sum(ũ.^2))
        sv = log10(ṽ[end]^2 / sum(ṽ.^2))
        isShock = max(detector(su, ps.deg), detector(sv, ps.deg))
        if isShock
            for s = 1:size(itg.u, 3)
                û = VInv * itg.u[i, :, s, :]
                #FR.filter_exp!(û, 10, 100)
                #FR.modal_filter!(û, 0.8e-2, 1e-6; filter = :l2)
                FR.modal_filter!(û, 10e-2, 5e-5; filter = :l2opt)
                #FR.modal_filter!(û, 5e-2, 5e-5; filter = :l2)
                #FR.modal_filter!(û, 5e-2, 5e-5; filter = :l2)
                itg.u[i, :, s, :] .= ps.V * û
            end
        end
    end
end

begin
    x = zeros(ncell * nsp)
    w = zeros(ncell * nsp, 3, uq.nm+1)
    for i = 1:ncell
        idx0 = (i - 1) * nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = ps.xpg[i, j]

            w[idx, :, :] .= itg.u[i, j, :, :]
        end
    end

    sol = zeros(ncell*nsp, 3, 2)
    for i in axes(sol, 1)
        p1 = zeros(3, uq.nm+1)
        p1 = uq_conserve_prim(w[i, :, :], γ, uq)
        p1[end, :] .= lambda_tchaos(p1[end, :], 1.0, uq)

        for k = 1:3
            sol[i, k, 1] = mean(p1[k, :], uq.op)
            sol[i, k, 2] = std(p1[k, :], uq.op)
        end
    end

    pic1 = plot(x, sol[:, 1, 1], label="ρ", xlabel="x", ylabel="mean")
    plot!(pic1, x, sol[:, 2, 1], label="U")
    plot!(pic1, x, sol[:, 3, 1], label="T")
    pic2 = plot(x, sol[:, 1, 2], label="ρ", xlabel="x", ylabel="std")
    plot!(pic2, x, sol[:, 2, 2], label="U")
    plot!(pic2, x, sol[:, 3, 2], label="T")
    plot(pic1, pic2)
end

plot(x, sol[:, 1, 2], label="adaptive", xlabel="x", ylabel="ρ")
