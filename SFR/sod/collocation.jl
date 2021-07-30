using KitBase, FluxReconstruction, OrdinaryDiffEq, Langevin, LinearAlgebra, Plots, JLD2
using ProgressMeter: @showprogress

begin
    x0 = 0
    x1 = 1
    ncell = 300
    nface = ncell + 1
    dx = (x1 - x0) / ncell
    deg = 2 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.05
    dt = cfl * dx / 2.0
    t = 0.0
end

ps = FRPSpace1D(x0, x1, ncell, deg)
VInv = inv(Array(ps.V))
ℓ = FR.basis_norm(ps.deg)

begin
    uqMethod = "collocation"
    nr = 9
    nRec = 18
    opType = "uniform"
    parameter1 = 0.9
    parameter2 = 1.1
end

uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)

u = zeros(ncell, nsp, 3, uq.nq)
begin
    isRandomLocation = true#false

    if isRandomLocation
        # stochastic location
        for i = 1:ncell, j = 1:nsp
            prim = zeros(3, uq.nq)

            for k = 1:uq.nq
                if ps.x[i] <= 0.5 + 0.05 * uq.op.quad.nodes[k]
                    prim[:, k] .= [1.0, 0.0, 0.5]
                else
                    prim[:, k] .= [0.125, 0.0, 0.625]
                end

                u[i, j, :, k] .= prim_conserve(prim[:, k], γ)
            end
        end
    else
        # stochastic density
        for i = 1:ncell, j = 1:nsp
            prim = zeros(3, uq.nq)
            for k = 1:uq.nq
                if ps.x[i] <= 0.5
                    prim[:, k] .= [uq.pceSample[k], 0.0, 0.5]
                else
                    prim[:, k] .= [0.125, 0.0, 0.625]
                end

                u[i, j, :, k] .= prim_conserve(prim[:, k], γ)
            end
        end
    end
end

function dudt!(du, u, p, t)
    du .= 0.0
    nx, nsp, nq, J, ll, lr, lpdm, dgl, dgr, γ = p

    ncell = size(u, 1)
    nsp = size(u, 2)

    f = zeros(ncell, nsp, 3, nq)
    for i = 1:ncell, j = 1:nsp, k = 1:nq
        f[i, j, :, k] .= euler_flux(u[i, j, :, k], γ)[1] ./ J[i]
    end

    u_face = zeros(ncell, 3, nq, 2)
    f_face = zeros(ncell, 3, nq, 2)
    for i = 1:ncell, j = 1:3, k = 1:nq
        # right face of element i
        u_face[i, j, k, 1] = dot(u[i, :, j, k], lr)
        f_face[i, j, k, 1] = dot(f[i, :, j, k], lr)

        # left face of element i
        u_face[i, j, k, 2] = dot(u[i, :, j, k], ll)
        f_face[i, j, k, 2] = dot(f[i, :, j, k], ll)
    end

    f_interaction = zeros(nx + 1, 3, nq)
    for i = 2:nx, j = 1:nq
        fw = @view f_interaction[i, :, j]
        flux_hll!(fw, u_face[i-1, :, j, 1], u_face[i, :, j, 2], γ, 1.0)
    end
    for j = 1:nq
        fw = @view f_interaction[1, :, j]
        flux_hll!(fw, u_face[nx, :, j, 1], u_face[1, :, j, 2], γ, 1.0)
        fw = @view f_interaction[nx+1, :, j]
        flux_hll!(fw, u_face[nx, :, j, 1], u_face[1, :, j, 2], γ, 1.0)
    end

    rhs1 = zero(u)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:3, l = 1:nq
        rhs1[i, ppp1, k, l] = dot(f[i, :, k, l], lpdm[ppp1, :])
    end

    idx = 2:ncell-1 # ending points are Dirichlet
    for i in idx, ppp1 = 1:nsp, k = 1:3, l = 1:nq
        du[i, ppp1, k, l] =
            -(
                rhs1[i, ppp1, k, l] +
                (f_interaction[i, k, l] / J[i] - f_face[i, k, l, 2]) * dgl[ppp1] +
                (f_interaction[i+1, k, l] / J[i] - f_face[i, k, l, 1]) * dgr[ppp1]
            )
    end
end

tspan = (0.0, 0.15)
p = (ps.nx, ps.deg + 1, uq.nq, ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, γ)
prob = ODEProblem(dudt!, u, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    for i = 1:ps.nx, j = 1:uq.nq
        tmp = @view itg.u[i, :, :, j]
        positive_limiter(tmp, γ, ps.wp ./ 2, ps.ll, ps.lr)
    end

    step!(itg)

    for i = 1:ps.nx, j = 1:uq.nq
        ũ = VInv * itg.u[i, :, 1, j]
        su = ũ[end]^2 / sum(ũ.^2)
        isShock = shock_detector(log10(su), ps.deg)

        if isShock
            for s = 1:size(itg.u, 3)
                λ = 5 * dt * sqrt(su)
                û = VInv * itg.u[i, :, s, j]
                #FR.modal_filter!(û, λ; filter = :l2)
                FR.modal_filter!(û, 4e-3; filter = :l2)
                #FR.modal_filter!(û, 2e-2; filter = :l2opt)
                #FR.modal_filter!(û, 1e-2, 1e-3; filter = :l2)
                #FR.modal_filter!(û, ℓ; filter = :lasso)

                itg.u[i, :, s, j] .= ps.V * û
            end
        end
    end
end

begin
    prim = zeros(ncell, nsp, 3, 2)
    for i in axes(prim, 1), j in axes(prim, 2)
        p0 = zeros(3, uq.nq)
        for k = 1:uq.nq
            p0[:, k] .= conserve_prim(itg.u[i, j, :, k], γ)
            p0[end, k] = 1 / p0[end, k]
        end
        
        p1 = zeros(3, uq.nm+1)
        for k = 1:3
            p1[k, :] .= ran_chaos(p0[k, :], uq)
        end

        for k = 1:3
            prim[i, j, k, 1] = mean(p1[k, :], uq.op)
            prim[i, j, k, 2] = std(p1[k, :], uq.op)
        end
    end

    x = zeros(ncell * nsp)
    sol = zeros(ncell * nsp, 3, 2)
    for i = 1:ncell
        idx0 = (i - 1) * nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = ps.xpg[i, j]

            sol[idx, :, :] .= prim[i, j, :, :]
        end
    end
end

plot(x, sol[:, 1, 1])
plot(x, sol[:, 1, 2])

#sol1 = deepcopy(sol)
#sol2 = deepcopy(sol)

#cd(@__DIR__)
@save "collocation.jld2" x sol1 sol2
