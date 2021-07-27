cd(@__DIR__)
include("tmp1.jl")

itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

ℓ = FR.basis_norm(ps.deg)

@showprogress for iter = 1:nt
    step!(itg)

    for i = 1:ps.nx
        ũ = VInv * itg.u[i, :, 1]
        su = ũ[end]^2 / sum(ũ.^2)
        isShock = shock_detector(log10(su), ps.deg)

        if isShock
            for s = 1:size(itg.u, 3)
                λ = dt * (su) * 5000
                û = VInv * itg.u[i, :, s]
                #FR.modal_filter!(û, λ; filter = :l2)
                #FR.modal_filter!(û, 1e-2; filter = :l2)
                #FR.modal_filter!(û, λ1, λ2; filter = :l2opt)
                #FR.modal_filter!(û, 1e-2, 1e-3; filter = :l2)
                FR.modal_filter!(û, ℓ; filter = :lasso)

                itg.u[i, :, s] .= ps.V * û
            end

            tmp = @view itg.u[i, :, :]
            positive_limiter(tmp, γ, ps.wp ./ 2, ps.ll, ps.lr)
            #slimiter(tmp, γ, ps.wp ./ 2, ps.ll, ps.lr)
        end

        #tmp = @view itg.u[i, :, :]
        #positive_limiter(tmp, γ, ps.wp ./ 2, ps.ll, ps.lr)
        #slimiter(tmp, γ, ps.wp ./ 2, ps.ll, ps.lr)
    end
end

begin
    x = zeros(ncell * nsp)
    sol = zeros(ncell * nsp, 3)
    for i = 1:ncell
        idx0 = (i - 1) * nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = ps.xpg[i, j]

            sol[idx, :] .= conserve_prim(itg.u[i, j, :], γ)
            sol[idx, 3] = 1 / sol[idx, 3]
        end
    end
    plot(x, sol[:, :])
end

#plot(x, sol[:, 1])

function tj_equation(t, p)
    ũ, u_mean, γ, ϵ = p
    
    u_temp = [
        t * (ũ[1] - u_mean[1]) + u_mean[1],
        t * (ũ[2] - u_mean[2]) + u_mean[2],
        t * (ũ[3] - u_mean[3]) + u_mean[3],
    ]
    prim_temp = conserve_prim(u_temp, γ)

    return 0.5 * prim_temp[1] / prim_temp[3] - ϵ
end

function positive_limiter(u::AbstractMatrix{T}, γ, weights, ll, lr) where {T<:AbstractFloat}
    u_mean = [sum(u[:, 1] .* weights), sum(u[:, 2] .* weights), sum(u[:, 3] .* weights)]
    prim_mean = conserve_prim(u_mean, γ)
    p_mean = 0.5 * prim_mean[1] / prim_mean[3]
    
    ϵ = min(1e-13, u_mean[1], p_mean)
    ρb = [dot(u[:, 1], ll), dot(u[:, 1], lr)]
    ρ_min = minimum(ρb)
    t1 = min((u_mean[1] - ϵ) / (u_mean[1] - ρ_min + 1e-8), 1.0)
    t1 = ifelse(t1 > 0, t1, 1.0)

    ρ̃ = zero(u[:, 1])
    for i in eachindex(ρ̃)
        ρ̃[i] = t1 * (u[i, 1] - prim_mean[1]) + prim_mean[1]
    end
    
    ũ = deepcopy(u)
    ũ[:, 1] .= ρ̃

    mb = [dot(u[:, 2], lr), dot(u[:, 2], ll)]
    eb = [dot(u[:, 3], lr), dot(u[:, 3], ll)]

    tj = Float64[]

    for i = 1:2
        prim = conserve_prim([ρb[i], mb[i], eb[i]], γ)
        pressure = 0.5 * prim[1] / prim[3]

        if pressure < ϵ
            prob = NonlinearProblem{false}(tj_equation, 1.0, ([ρb[i], mb[i], eb[i]], u_mean, γ, ϵ))
            sol = solve(prob, NewtonRaphson(), tol = 1e-6)
            push!(tj, sol.u)
        end
    end

    if length(tj) == 0
        u .= ũ
    else
        t2 = minimum(tj)

        for i in axes(ũ, 1)
            @. u[i, :] = t2 * (ũ[i, :] - u_mean[:]) + u_mean[:]
        end
    end

    return nothing
end
