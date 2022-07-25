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
    dt = cfl * dx / 2
    t = 0.0
end
ps = FRPSpace1D(x0, x1, ncell, deg)
VInv = inv(Array(ps.V))
ℓ = FR.basis_norm(ps.deg)

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
    end

    f_interaction = zeros(nx + 1, 3)
    for i = 2:nx
        fw = @view f_interaction[i, :]
        flux_hll!(fw, u_face[i-1, :, 1], u_face[i, :, 2], γ, 1.0)
    end

    rhs1 = zeros(ncell, nsp, 3)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:3
        rhs1[i, ppp1, k] = dot(f[i, :, k], lpdm[ppp1, :])
    end

    idx = 2:ncell-1 # ending points are Dirichlet
    for i in idx, ppp1 = 1:nsp, k = 1:3
        du[i, ppp1, k] = -(
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
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    for i = 1:ps.nx
        ũ = VInv * itg.u[i, :, 1]
        su = ũ[end]^2 / sum(ũ .^ 2)
        isShock = shock_detector(log10(su), ps.deg)

        if isShock
            for s = 1:size(itg.u, 3)
                λ = 5 * dt * sqrt(su) #/ (ps.xpg[i, 2] - ps.xpg[i, 1])
                #λ = 5 * dt * max(abs(dot(itg.u[i, :, 1], ps.dl[1, :])), abs(dot(itg.u[i, :, 1], ps.dl[2, :]))) / ps.J[i] #/ itg.u[i, 1, 1]
                û = VInv * itg.u[i, :, s]
                #FR.modal_filter!(û, λ; filter = :l2)
                #FR.modal_filter!(û, 1.5e-2; filter = :l2)
                #FR.modal_filter!(û, 2e-2; filter = :l2opt)
                #FR.modal_filter!(û, 1e-2, 1e-3; filter = :l2)
                FR.modal_filter!(û, ℓ; filter = :lasso)

                itg.u[i, :, s] .= ps.V * û
            end
        end

        tmp = @view itg.u[i, :, :]
        positive_limiter(tmp, γ, ps.wp ./ 2, ps.ll, ps.lr)
    end

    step!(itg)
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

#plot(x, sol[:, 2])
