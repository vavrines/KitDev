using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots, NonlinearSolve
using ProgressMeter: @showprogress

begin
    x0 = 0
    x1 = 1
    ncell = 100
    dx = (x1 - x0) / ncell
    deg = 2 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.05
    dt = cfl * dx
    t = 0.0
end
ps = FRPSpace1D(x0, x1, ncell, deg)
VInv = inv(Array(ps.V))

u = zeros(ncell, nsp, 3)
for i = 1:ncell, ppp1 = 1:nsp
    if ps.x[i] <= 0.5
        prim = [1.0, 0.0, 0.5]
    else
        prim = [0.125, 0.0, 0.625]
    end

    u[i, ppp1, :] .= prim_conserve(prim, γ)
end

function dudt!(du, u, p, t)
    du .= 0.0
    nx, nsp, J, ll, lr, lpdm, dgl, dgr, γ = p

    for i = 1:nx, j = 1:nsp
        prim = conserve_prim(u[i, j, :], γ)
        if prim[1] <= 0 || prim[end] <= 0
            @show i, j
        end
    end

    ncell = size(u, 1)
    nsp = size(u, 2)

    f = zeros(ncell, nsp, 3)
    for i = 1:ncell, j = 1:nsp
        f[i, j, :] .= euler_flux(u[i, j, :], γ)[1] ./ J[i]
    end

    u_face = zeros(ncell, 3, 2)
    f_face = zeros(ncell, 3, 2)
    for i = 1:ncell
        for j = 1:3
            # right face of element i
            u_face[i, j, 1] = dot(u[i, :, j], lr)
            f_face[i, j, 1] = dot(f[i, :, j], lr)

            # left face of element i
            u_face[i, j, 2] = dot(u[i, :, j], ll)
            f_face[i, j, 2] = dot(f[i, :, j], ll)
        end
#=
        prim1 = conserve_prim(u_face[i, :, 1], γ)
        if prim1[1] < 0 || prim1[3] < 0
            u_face[i, :, 1] .= u[i, nsp, :]
            f_face[i, :, 1] .= f[i, nsp, :]
        end
        prim2 = conserve_prim(u_face[i, :, 2], γ)
        if prim2[1] < 0 || prim2[3] < 0
            u_face[i, :, 2] .= u[i, 1, :]
            f_face[i, :, 2] .= f[i, 1, :]
        end=#

        #=if prim1[1] < 0 || prim1[3] < 0 || prim2[1] < 0 || prim2[3] < 0
            tmp = @view u[i, :, :]
            slimiter(tmp, γ, ps.wp ./ 2, ll, lr)
            for j in axes(f, 2)
                f[i, j, :] .= euler_flux(u[i, j, :], γ)[1] ./ J[i]
            end
        end

        for j = 1:3
            # right face of element i
            u_face[i, j, 1] = dot(u[i, :, j], lr)
            f_face[i, j, 1] = dot(f[i, :, j], lr)

            # left face of element i
            u_face[i, j, 2] = dot(u[i, :, j], ll)
            f_face[i, j, 2] = dot(f[i, :, j], ll)
        end
=#
    end

    f_interaction = zeros(nx + 1, 3)
    for i = 2:nx
        fw = @view f_interaction[i, :]
        flux_hll!(fw, u_face[i-1, :, 1], u_face[i, :, 2], γ, 1.0)
    end
    #fw = @view f_interaction[1, :]
    #flux_hll!(fw, u_face[nx, :, 1], u_face[1, :, 2], γ, 1.0)
    #fw = @view f_interaction[nx+1, :]
    #flux_hll!(fw, u_face[nx, :, 1], u_face[1, :, 2], γ, 1.0)

    rhs1 = zeros(ncell, nsp, 3)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:3
        rhs1[i, ppp1, k] = dot(f[i, :, k], lpdm[ppp1, :])
    end

    idx = 2:ncell-1 # ending points are Dirichlet
    for i in idx, ppp1 = 1:nsp, k = 1:3
        du[i, ppp1, k] =
            -(
                rhs1[i, ppp1, k] +
                (f_interaction[i, k] / J[i] - f_face[i, k, 2]) * dgl[ppp1] +
                (f_interaction[i+1, k] / J[i] - f_face[i, k, 1]) * dgr[ppp1]
            )
    end
end

tspan = (0.0, 0.15)
p = (ps.nx, ps.deg + 1, ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, γ)
prob = ODEProblem(dudt!, u, tspan, p)
nt = tspan[2] ÷ dt |> Int




function slimiter(u::AbstractMatrix{T}, γ, weights, ll, lr) where {T<:AbstractFloat}
    prim = deepcopy(u)
    for i in axes(u, 1)
        prim[i, :] .= conserve_prim(u[i, :], γ)
    end
    prim_mean = [
        sum(prim[:, 1] .* weights),
        sum(prim[:, 2] .* weights),
        sum(prim[:, 3] .* weights),
    ]
    ρ_mean = prim_mean[1]
    t_mean = 1 / prim_mean[3]
    p_mean = 0.5 * ρ_mean * t_mean

    ρb = [dot(prim[:, 1], ll), dot(prim[:, 1], lr)]
    tb = [dot(1 ./ prim[:, 3], ll), dot(1 ./ prim[:, 3], lr)]
    ρ_min = minimum(ρb)
    t_min = minimum(tb)

    if true#ρ_min < 0 || t_min < 0
        ϵ = min(1e-6, ρ_mean, t_mean)
        t1 = min((ρ_mean - ϵ) / (ρ_mean - ρ_min + 1e-6), 1.0) |> abs
        t2 = min((t_mean - ϵ) / (t_mean - t_min + 1e-6), 1.0) |> abs
        
        #@assert t1 >= 0 "ρ_mean =$(u_mean[1]), ϵ=$ϵ, ρ_min = $ρ_min"
        #@assert t2 >= 0 "t_mean =$(t_mean), ϵ=$ϵ, t_min = $t_min"

        t = 0.97#min(t1, t2)
        if t < 1.0
            @show t1 t2
            @show t ϵ
            @show ρ_mean u[:, 1] ρb[1] ρb[2]
            @show t_mean tb[1] tb[2]
        end
        prim_tilde = deepcopy(prim)
        for i in axes(u, 1)
            prim_tilde[i, 1] = t * (prim[i, 1] - ρ_mean) + ρ_mean

            prim_tilde[i, 2] = t * (prim[i, 2] - prim_mean[2]) + prim_mean[2]

            tmp = t * (1 / prim[i, 3] - t_mean) + t_mean
            prim_tilde[i, 3] = 1 / tmp
        end

        for i in axes(u, 1)
            u[i, :] .= prim_conserve(prim_tilde[i, :], γ)
        end
    end

    return nothing
end

#=
function slimiter(u::AbstractMatrix{T}, γ, weights, ll, lr) where {T<:AbstractFloat}
    prim = deepcopy(u)
    for i in axes(u, 1)
        prim[i, :] .= conserve_prim(u[i, :], γ)
    end

    







    ρ = u[:, 1]
    e = @. u[:, 3] - 0.5 * u[:, 2]^2 / u[:, 1]      

    ρb = [dot(ρ, ll), dot(ρ, lr)]
    eb = [dot(e, ll), dot(e, lr)]

    ρmin = minimum(ρb)
    emin = minimum(eb)
    tmin = minimum(eb ./ ρb)
    
    
    u_mean = [sum(u[:, 1] .* weights), sum(u[:, 2] .* weights), sum(u[:, 3] .* weights)]

    ϵ = min(1e-12, u_mean[1], u_mean[3])

    t1 = min((u_mean[1] - ϵ) / (u_mean[1] - ρmin + 1e-6), 1.0)
    t2 = min((u_mean[3] - ϵ) / (u_mean[3] - emin + 1e-6), 1.0)
    t = min(t1, t2)

    t < 1 && @show t

    for i in axes(u, 1)
        u[i, 1] = t * (u[i, 1] - u_mean[1]) + u_mean[1]
        u[i, 2] = t * (u[i, 2] - u_mean[2]) + u_mean[2]
        u[i, 3] = t * (u[i, 3] - u_mean[3]) + u_mean[3]
    end

    return nothing
end=#

