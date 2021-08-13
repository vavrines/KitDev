using KitBase, FluxReconstruction, OrdinaryDiffEq, Langevin, LinearAlgebra, Plots, JLD2
using ProgressMeter: @showprogress

function FR.positive_limiter(u::AbstractArray{T,3}, γ, wp, wq, ll, lr, t0 = 1.0) where {T<:AbstractFloat}
    # tensorized quadrature weights
    weights = zeros(length(wp), length(wq))
    for i in axes(weights, 1), j in axes(weights, 2)
        weights[i, j] = wp[i] * wq[j]
    end
    
    # mean values
    u_mean = [sum(u[:, j, :] .* weights) for j in axes(u, 2)]
    t_mean = 1.0 / conserve_prim(u_mean, γ)[end]
    @assert t_mean > 0

    # boundary variables
    ρb = zeros(2, length(wq))
    mb = zero(ρb)
    eb = zero(ρb)
    for j in axes(ρb, 2)
        ρb[:, j] .= [dot(u[:, 1, j], ll), dot(u[:, 1, j], lr)]
        mb[:, j] .= [dot(u[:, 2, j], ll), dot(u[:, 2, j], lr)]
        eb[:, j] .= [dot(u[:, 3, j], ll), dot(u[:, 3, j], lr)]
    end

    # density corrector
    ϵ = min(1e-13, u_mean[1], t_mean)
    ϵ < 0 && @show ϵ
    ρ_min = min(minimum(ρb), minimum(u[:, 1, :])) # density minumum can emerge at both solution and flux points
    t1 = min((u_mean[1] - ϵ) / (u_mean[1] - ρ_min + 1e-8), 1.0)
    @assert 0 <= t1 <= 1 "incorrect range of limiter parameter t"
    #t1 < 1 && @show t1

    for i in axes(u, 1), j in axes(u, 3)
        u[i, 1, j] = t1 * (u[i, 1, j] - u_mean[1]) + u_mean[1]
    end

    # energy corrector
    tj = Float64[]
    for i = 1:2
        prim = conserve_prim([ρb[i], mb[i], eb[i]], γ)

        if 1 / prim[end] < ϵ
            prob = NonlinearProblem{false}(FR.tj_equation, 1.0, ([ρb[i], mb[i], eb[i]], u_mean, γ, ϵ))
            sol = solve(prob, FR.NonlinearSolve.NewtonRaphson(), tol = 1e-9)
            push!(tj, sol.u)
        end
    end
    for i in axes(u, 1), j in axes(u, 3)
        prim = conserve_prim(u[i, :, j], γ)

        if 1 / prim[end] < ϵ
            prob = NonlinearProblem{false}(FR.tj_equation, 1.0, (u[i, :, j], u_mean, γ, ϵ))
            sol = solve(prob, FR.NonlinearSolve.NewtonRaphson(), tol = 1e-9)
            push!(tj, sol.u)
        end
    end

    if length(tj) > 0
        t2 = min(minimum(tj), t0)
        #t2 < 1 && @show t2
        @assert 0 <= t2 <= 1 "incorrect range of limiter parameter t"
        for k in axes(u, 3), j in axes(u, 2), i in axes(u, 1)
            u[i, j, k] = t2 * (u[i, j, k] - u_mean[j]) + u_mean[j]
        end
    end
    
    #=for j in axes(ρb, 2)
        ρb[:, j] .= [dot(u[:, 1, j], ll), dot(u[:, 1, j], lr)]
        eb[:, j] .= [dot(u[:, 3, j], ll), dot(u[:, 3, j], lr)]
    end
    if minimum(eb) <= 0
        for k in axes(u, 3), i in axes(u, 1)
            u[i, :, k] .= u_mean
        end
    end=#

    return nothing
end

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
    parameter1 = 0.9
    parameter2 = 1.1
end
ps = FRPSpace1D(x0, x1, ncell, deg)
uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)
V = vandermonde_matrix(ps.deg,ps.xpl)
VInv = inv(Array(ps.V))
l2 = [uq.t2Product[j-1, j-1] for j = 1:uq.nm+1]
#l2 = [2 / (2 * (i-1) + 1) for i = 1:uq.nm+1]

cd(@__DIR__)
include("rhs.jl")
include("../filter.jl")

u = zeros(ncell, nsp, 3, uq.nm+1)
case = ("location", nothing)[1]

if case == "location"
    for i = 1:ncell, j = 1:nsp
        prim = zeros(3, uq.nq)

        for k = 1:uq.nq
            if ps.x[i] <= 0.5 + 0.05 * uq.op.quad.nodes[k]
            #if ps.xpg[i, j] <= 0.5 + 0.05 * uq.op.quad.nodes[k] 
                prim[:, k] .= [1.0, 0.0, 0.5]
            else
                prim[:, k] .= [0.125, 0.0, 0.625]
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
            #FR.modal_filter!(uModal, 10, 10; filter = :exp)
            #FR.modal_filter!(uModal; filter = :lasso)
            #FR.modal_filter!(uModal, 1e-3, 1e-3; filter = :l2opt)
            FR.modal_filter!(uModal, 1e-5, 1e-4; filter = :l2)
            #FR.modal_filter!(uModal, 5e-2, 2e-4; filter = :l2)
            u[j, :, s, :] .= V * uModal
        end
    end
else
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

p = (ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, γ, uq)
prob = ODEProblem(dudt!, u, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    #=for i = 1:size(itg.u, 1)
        uNodal = zeros(nsp, 3, uq.nq)
        for idx = 1:nsp, jdx = 1:3
            uNodal[idx, jdx, :] .= chaos_ran(itg.u[i, idx, jdx, :], uq)
        end
        positive_limiter(uNodal, γ, ps.wp/2, uq.op.quad.weights, ps.ll, ps.lr, 0.99)
        #=for idx = 1:nsp
            tmp = @view uNodal[idx, :, :]
            positive_limiter(tmp, γ, uq.op.quad.weights)
        end=#
        for idx = 1:nsp, jdx = 1:3
            itg.u[i, idx, jdx, :] .= ran_chaos(uNodal[idx, jdx, :], uq)
        end
    end=#

    step!(itg)

    for i = 1:size(itg.u, 1)
        ũ = VInv * itg.u[i, :, 1, :]
        #su = maximum([ũ[end, j]^2 / sum(ũ[:, j].^2) for j = 1:uq.nm+1])
        #=su = maximum([ũ[end, j]^2 / (sum(ũ[:, j].^2) + 1e-6) for j = 1:uq.nm+1])
        sv = maximum([ũ[j, end]^2 * l2[end] / (sum(ũ[j, :].^2 .* l2) + 1e-6) for j = 1:nsp])
        isShock = max(
            shock_detector(log10(su), ps.deg, -3 * log10(ps.deg), 4.0),
            shock_detector(log10(sv), ps.deg, -3 * log10(ps.deg), 4.0),
        )=#

        su = (ũ[end, end]^2 + ũ[end-1, end]^2 + ũ[end, end-1]^2) / (sum(ũ.^2) + 1e-6)
        isShock = max(
            shock_detector(log10(su), ps.deg, -3 * log10(ps.deg), 4.0),
        )

        su = ũ[end, 1]^2 / (sum(ũ[:, 1].^2) + 1e-6)
        sv = ũ[1, end]^2 * l2[end] / (sum(ũ[1, :].^2 .* l2) + 1e-6)

        if isShock
            #λ1 = 2e-3
            #λ2 = 1e-5
            λ1 = dt * exp(0.875/1 * (ps.deg+1)) * 1.2
            λ2 = λ1 / 225

            #λ1 = sqrt(su) * 10.0#20
            #λ2 = sqrt(sv) * 1#10
            #λ1 = 5e-2
            #λ2 = 5e-4

            for s = 1:size(itg.u, 3)
                û = VInv * itg.u[i, :, s, :]
                #FR.modal_filter!(û, λ1, λ2; filter = :l2)
                #FR.modal_filter!(û, λ1, λ2; filter = :l2opt)
                #FR.modal_filter!(û; filter = :lasso)
                
                #FR.modal_filter!(û, 4.5; filter = :exp)
                FR.modal_filter!(û, 8, 7; filter = :exp)

                #FR.modal_filter!(û, 2e-3, 1e-5; filter = :l2)
                #FR.modal_filter!(û, 5e-2, 1e-2; filter = :l2opt)

                #FR.modal_filter!(û, 2e-3, 1e-5; filter = :l2)
                
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

plot(x, sol[:, 1, 1])
plot(x, sol[:, 1, 2])

#x0 = deepcopy(x); sol0 = deepcopy(sol)

#sol1 = deepcopy(sol)
#sol2 = deepcopy(sol)

#@save "lasso.jld2" x sol1 sol2
#@save "l2.jld2" x sol1 sol2
#@save "l2_apt.jld2" x sol1 sol2