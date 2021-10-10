using OrdinaryDiffEq, LinearAlgebra, Plots, Statistics
using KitBase, FluxReconstruction
using ProgressMeter: @showprogress

begin
    x0 = 0
    x1 = 1
    nx = 50
    deg = 2
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 100
    knudsen = 1e-2
end

set = Setup(
    case = "sod",
    space = "1d1f1v",
    boundary = "fix",
    cfl = 0.2,
    maxTime = 0.1,
)

ps = FRPSpace1D(x0, x1, nx, deg)
vs = VSpace1D(u0, u1, nu)
δ = heaviside.(vs.u)

function mol!(du, u, p, t)
    dx, velo, weights, δ, μ, ll, lr, lpdm, dgl, dgr = p

    ncell = length(dx)
    nu = length(velo)
    nsp = length(ll)

    f = zero(u)
    @inbounds Threads.@threads for i = 1:ncell
        J = 0.5 * dx[i]

        for k = 1:nsp
            @. f[i, :, k] = velo * u[i, :, k] / J
        end
    end

    u_face = zeros(eltype(u), ncell, nu, 2)
    f_face = zeros(eltype(u), ncell, nu, 2)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:nu, k = 1:nsp
            # right face of element i
            u_face[i, j, 1] += u[i, j, k] * lr[k]
            f_face[i, j, 1] += f[i, j, k] * lr[k]

            # left face of element i
            u_face[i, j, 2] += u[i, j, k] * ll[k]
            f_face[i, j, 2] += f[i, j, k] * ll[k]
        end
    end

    u_interaction = zeros(eltype(u), ncell+1, nu)
    f_interaction = zeros(eltype(u), ncell+1, nu)
    @inbounds Threads.@threads for i = 2:ncell
        @. u_interaction[i, :] = u_face[i, :, 2] * (1.0 - δ) + u_face[i-1, :, 1] * δ
        @. f_interaction[i, :] = f_face[i, :, 2] * (1.0 - δ) + f_face[i-1, :, 1] * δ
    end

    rhs = zeros(eltype(u), ncell, nu, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:nu, ppp1 = 1:nsp, k = 1:nsp
            rhs[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for i = 2:ncell-1
        for ppp1 = 1:nsp
            cons = moments_conserve(u[i, :, ppp1], velo, weights)
            M = maxwellian(velo, conserve_prim(cons, 3))
            τ = vhs_collision_time(conserve_prim(cons, 3), μ, 0.81)

            j = 1:nu
            du[i, j, ppp1] .=
                -(
                    rhs[i, j, ppp1] .+
                    (f_interaction[i, j] .- f_face[i, j, 2]) .* dgl[ppp1] .+
                    (f_interaction[i+1, j] .- f_face[i, j, 1]) .* dgr[ppp1]
                ) .+ (M .- u[i, j, ppp1]) ./ τ
        end
    end

    du[1, :, :] .= 0.0
    du[ncell, :, :] .= 0.0
end

begin
    u0 = zeros(nx, nu, nsp)
    for i = 1:nx, ppp1 = 1:nsp
        if i <= nx÷2
            _ρ = 1.0
            _λ = 0.5
        else
            _ρ = 0.125
            _λ = 0.625
        end

        u0[i, :, ppp1] .= maxwellian(vs.u, [_ρ, 0.0, _λ])
    end
end

tspan = (0.0, set.maxTime)
dt = set.cfl * minimum(ps.dx) / (vs.u1 + 2)
nt = floor(tspan[2] / dt) |> Int
p = (ps.dx, vs.u, vs.weights, δ, ref_vhs_vis(knudsen, 1.0, 0.5), 
    ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr)

prob = ODEProblem(mol!, u0, tspan, p)
itg = init(
    prob,
    Midpoint(),
    #reltol = 1e-8,
    #abstol = 1e-8,
    save_everystep = false,
    adaptive = false,
    dt = dt,
)

@showprogress for iter = 1:nt
    step!(itg)
end

begin
    x = zeros(nx * nsp)
    w = zeros(nx * nsp, 3)
    prim = zeros(nx * nsp, 4)
    for i = 1:nx
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = ps.xpg[i, j]

            w[idx, :] = moments_conserve(itg.u[i, :, j], vs.u, vs.weights)
            prim[idx, 1:3] .= conserve_prim(w[idx, :], 3)
            prim[idx, 4] = 0.5 * prim[idx, 1] / prim[idx, 3]
        end
    end
end

plot(x[1:end], markeralpha=0.6, prim[1:end, 1:2])
plot!(x[1:end], markeralpha=0.6, prim[1:end, 4])
