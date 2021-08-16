using KitBase, FluxReconstruction, OrdinaryDiffEq, Langevin, LinearAlgebra, Plots, JLD2
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

begin
    x0 = 0
    x1 = 3
    ncell = 100#300
    nface = ncell + 1
    dx = (x1 - x0) / ncell
    deg = 3 # polynomial degree
    nsp = deg + 1
    cfl = 0.05
    dt = cfl * dx / 10
    t = 0.0

    uqMethod = "galerkin"
    nr = 9
    nRec = 18
    opType = "uniform"
    parameter1 = -1.0
    parameter2 = 1.0
end

ps = FRPSpace1D(x0, x1, ncell, deg)
uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)
VInv = inv(Array(ps.V))
l2 = [uq.t2Product[j-1, j-1] for j = 1:uq.nm+1]

cd(@__DIR__)
include("../filter.jl")

function l2_strength(a1, a2=a1)
    (1/a1 - 1)/2/ps.deg^2 / (ps.deg+1)^2, (1/a2-1)/2/uq.nm^2 / (uq.nm+1)^2
end

u0 = zeros(ncell, nsp, uq.nm+1)
u0q = zeros(ncell, nsp, uq.nq)
let p0 = 0.5, p1 = 1.5, σ = 0.2, val0 = 11.0, val1 = 1.0
    for i = 1:ncell
        for j = 1:uq.nq
            if ps.x[i] < p0 + σ * uq.pceSample[j]
                u0q[i, :, j] .= val0
            elseif p0 + σ * uq.pceSample[j] <= ps.x[i] <= p1 + σ * uq.pceSample[j]
                u0q[i, :, j] .= val0 + (val1 - val0) / (p0 - p1) * (p0 + σ * uq.pceSample[j] - ps.x[i])
            else
                u0q[i, :, j] .= val1
            end
        end

        for j = 1:nsp
            u0[i, j, :] .= ran_chaos(u0q[i, j, :], uq)
        end
    end
end

function dudt!(du, u, p, t)
    du .= 0.0
    J, ll, lr, lpdm, dgl, dgr, Δx, uq = p

    nm = uq.nm
    nq = uq.nq

    ncell = size(u, 1)
    nsp = size(u, 2)

    u_ran = zeros(ncell, nsp, nq)
    @threads for j = 1:nsp
        for i = 1:ncell
            @inbounds u_ran[i, j, :] .= chaos_ran(u[i, j, :], uq)
        end
    end

    f = zeros(ncell, nsp, nm+1)
    @threads for j = 1:nsp
        for i = 1:ncell
            _f = zeros(nq)
            for k = 1:nq
                _f[k] = 0.5 * u_ran[i, j, k]^2 / J[i]
            end

            f[i, j, :] .= ran_chaos(_f, uq)
        end
    end

    u_face = zeros(ncell, nm+1, 2)
    f_face = zeros(ncell, nm+1, 2)
    @inbounds @threads for k = 1:nm+1
        for i = 1:ncell
            u_face[i, k, 1] = dot(u[i, :, k], lr)
            f_face[i, k, 1] = dot(f[i, :, k], lr)
            u_face[i, k, 2] = dot(u[i, :, k], ll)
            f_face[i, k, 2] = dot(f[i, :, k], ll)
        end
    end

    f_interaction = zeros(ncell + 1, nm+1)
    @threads for i = 2:ncell
        @inbounds f_interaction[i, :] .= (f_face[i-1, :, 1] .+ f_face[i, :, 2]) ./ 2 - 
            (Δx[i-1] + Δx[i]) / 2 * (u_face[i, :, 2] .- u_face[i-1, :, 1])
    end
    #f_interaction[1, :] .= (f_face[ncell, :, 1] .+ f_face[1, :, 2]) ./ 2 - 
    #    (Δx[ncell] + Δx[1]) / 2 * (u_face[1, :, 2] .- u_face[ncell, :, 1])
    #f_interaction[ncell+1, :] .= (f_face[ncell, :, 1] .+ f_face[1, :, 2]) ./ 2 - 
    #    (Δx[ncell] + Δx[1]) / 2 * (u_face[1, :, 2] .- u_face[ncell, :, 1])

    rhs1 = zero(u)
    for i = 1:ncell, ppp1 = 1:nsp, l = 1:nm+1
        @inbounds rhs1[i, ppp1, l] = dot(f[i, :, l], lpdm[ppp1, :])
    end

    idx = 2:ncell-1
    @threads for l = 1:nm+1
        for i in idx, ppp1 = 1:nsp
            @inbounds du[i, ppp1, l] =
                -(
                    rhs1[i, ppp1, l] +
                    (f_interaction[i, l] - f_face[i, l, 2]) * dgl[ppp1] +
                    (f_interaction[i+1, l] - f_face[i, l, 1]) * dgr[ppp1]
                )
        end
    end
end

tspan = (0.0, 0.1)
p = (ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, ps.dx, uq)
prob = ODEProblem(dudt!, u0, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Tsit5(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)

    # filter
    for i = 1:size(itg.u, 1)
        ũ = VInv * itg.u[i, :, :]
        #su = maximum([ũ[end, j]^2 / sum(ũ[:, j].^2) for j = 1:uq.nm+1])
        #sv = maximum([ũ[j, end]^2 / sum(ũ[j, :].^2) for j = 1:nsp])
        #isShock = max(detector(log10(su), ps.deg), detector(log10(sv), ps.deg))
        #=if isShock
            λ1 = dt * (su)
            λ2 = dt * (sv)

            #FR.filter_exp!(uModal, 10, 100)
            FR.modal_filter!(ũ, λ1, λ2; filter = :l2)
            #FR.modal_filter!(ũ, λ1, λ2; filter = :l2opt)
            #FR.modal_filter!(ũ; filter = :lasso)

            itg.u[i, :, :] .= ps.V * ũ
        end=#

        su = maximum([ũ[end, j]^2 / (sum(ũ[:, j].^2) + 1e-4) for j = 1:uq.nm+1])
        sv = maximum([ũ[j, end]^2 * uq.t2Product[uq.nm, uq.nm] / (sum(ũ[j, :].^2 .* l2) + 1e-4) for j = 1:nsp])
        isShock = max(
            #shock_detector(log10(su), ps.deg, -3 * log10(ps.deg), 1.0),
            shock_detector(log10(sv), ps.deg, -2 * log10(ps.deg), 1.0),
        )

        if isShock
            #FR.modal_filter!(ũ, 16; filter = :exp)

            #FR.modal_filter!(ũ, 8, 8; filter = :houli)

            #λ1 = dt * sqrt(su) * 2.0
            #λ2 = dt * sqrt(sv) * 2.0

            #λ1 = 3.5e-2
            #λ2 = λ1 * ps.deg^2 * (ps.deg+1)^2 / uq.nm^2 / (uq.nm+1)^2

            λ1, λ2 = l2_strength(0.6)

            FR.modal_filter!(ũ, λ1, λ2; filter = :l2)
            #FR.modal_filter!(ũ, λ1, λ2; filter = :l2opt)
            #FR.modal_filter!(ũ, PhiL1; filter = :lasso)
            #FR.modal_filter!(ũ, 8, 8; filter = :exp)


            itg.u[i, :, :] .= ps.V * ũ
        end
    
        #=tmp = [chaos_ran(itg.u[i, j, :], uq) for j = 1:nsp]
        uquad = [tmp[i][j] for i=1:nsp, j=1:uq.nq]
        δu = [maximum(uquad[:, j]) - minimum(uquad[:, j]) for j = 1:uq.nm+1] |> maximum
        δv = [maximum(uquad[j, :]) - minimum(uquad[j, :]) for j = 1:nsp] |> maximum
        λ1 = dt * relu(δu - 7.0)
        λ2 = dt * relu(δv - 7.0)
        uModal = VInv * itg.u[i, :, :]
        FR.modal_filter!(uModal, λ1, λ2; filter = :l2)
        itg.u[i, :, :] .= ps.V * uModal=#
    end
end

begin
    x = zeros(ncell * nsp)
    w = zeros(ncell * nsp, uq.nm+1)
    for i = 1:ncell
        idx0 = (i - 1) * nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = ps.xpg[i, j]

            w[idx, :] .= itg.u[i, j, :]
        end
    end

    sol = zeros(ncell*nsp, 2)
    for i in axes(sol, 1)
        sol[i, 1] = mean(w[i, :], uq.op)
        sol[i, 2] = std(w[i, :], uq.op)
    end

    pic1 = plot(x, sol[:, 1], label="u", xlabel="x", ylabel="mean")
    pic2 = plot(x, sol[:, 2], label="u", xlabel="x", ylabel="std")
    plot(pic1, pic2)
end

plot(x, sol0[:, 2], label="Lasso", line=:dash)
plot!(x, sol[:, 2], label="L2", xlabel="x", ylabel="std")

uξ = chaos_ran(itg.u[53, 2, :], uq)
plot(uq.op.quad.nodes, uξ)

sol0 = deepcopy(sol)
#@save "nofilter.jld2" x sol
#@save "l2_apt.jld2" x sol
#@save "lasso_apt.jld2" x sol
#@save "l2opt_apt.jld2" x sol
#@save "exp.jld2" x sol
#@save "houli.jld2" x sol
#@save "lasso.jld2" x sol
#@save "l2.jld2" x sol
#@save "l2_apt.jld2" x sol
