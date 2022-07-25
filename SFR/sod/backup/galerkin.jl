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
    dt = cfl * dx
    t = 0.0
end

ps = FRPSpace1D(x0, x1, ncell, deg)

begin
    uqMethod = "galerkin"
    nr = 5
    nRec = 10
    opType = "uniform"
    parameter1 = 0.95
    parameter2 = 1.05
end

uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)

u = zeros(ncell, nsp, 3, uq.nm + 1)
for i = 1:ncell, j = 1:nsp
    prim = zeros(3, uq.nm + 1)
    if ps.x[i] <= 0.5
        prim[1, :] .= uq.pce
        prim[2, 1] = 0.0
        prim[3, 1] = 0.5
    else
        prim[:, 1] .= [0.3, 0.0, 0.625]
    end

    u[i, j, :, :] .= uq_prim_conserve(prim, γ, uq)
end

function dudt!(du, u, p, t)
    du .= 0.0
    J, ll, lr, lpdm, dgl, dgr, γ, uq = p

    nm = uq.nm
    nq = uq.nq

    ncell = size(u, 1)
    nsp = size(u, 2)

    u_ran = zeros(ncell, nsp, 3, nq)
    for i = 1:ncell, j = 1:nsp, k = 1:3
        u_ran[i, j, k, :] .= chaos_ran(u[i, j, k, :], uq)
    end

    f = zeros(ncell, nsp, 3, nm + 1)
    for i = 1:ncell, j = 1:nsp
        _f = zeros(3, nq)
        for k = 1:nq
            _f[:, k] .= euler_flux(u_ran[i, j, :, k], γ)[1] ./ J[i]
        end

        for k = 1:3
            f[i, j, k, :] .= ran_chaos(_f[k, :], uq)
        end
    end

    f_face = zeros(ncell, 3, nm + 1, 2)
    for i = 1:ncell, j = 1:3, k = 1:nm+1
        f_face[i, j, k, 1] = dot(f[i, :, j, k], lr)
        f_face[i, j, k, 2] = dot(f[i, :, j, k], ll)
    end

    u_face = zeros(ncell, 3, nq, 2)
    for i = 1:ncell, j = 1:3, k = 1:nq
        u_face[i, j, k, 1] = dot(u_ran[i, :, j, k], lr)
        u_face[i, j, k, 2] = dot(u_ran[i, :, j, k], ll)
    end

    fq_interaction = zeros(ncell + 1, 3, nq)
    for i = 2:ncell, j = 1:nq
        fw = @view fq_interaction[i, :, j]
        flux_hll!(fw, u_face[i-1, :, j, 1], u_face[i, :, j, 2], γ, 1.0)
    end

    f_interaction = zeros(ncell + 1, 3, nm + 1)
    for i = 2:ncell, j = 1:3
        f_interaction[i, j, :] .= ran_chaos(fq_interaction[i, j, :], uq)
    end

    rhs1 = zero(u)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:3, l = 1:nm+1
        rhs1[i, ppp1, k, l] = dot(f[i, :, k, l], lpdm[ppp1, :])
    end

    idx = 2:ncell-1 # ending points are Dirichlet
    for i in idx, ppp1 = 1:nsp, k = 1:3, l = 1:nm+1
        du[i, ppp1, k, l] = -(
            rhs1[i, ppp1, k, l] +
            (f_interaction[i, k, l] / J[i] - f_face[i, k, l, 2]) * dgl[ppp1] +
            (f_interaction[i+1, k, l] / J[i] - f_face[i, k, l, 1]) * dgr[ppp1]
        )
    end
end

tspan = (0.0, 0.15)
p = (ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, γ, uq)
prob = ODEProblem(dudt!, u, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)
end

sol = zeros(ncell, nsp, 3, 2)
for i in axes(sol, 1), j in axes(sol, 2)
    p1 = zeros(3, uq.nm + 1)
    p1 = uq_conserve_prim(itg.u[i, j, :, :], γ, uq)
    p1[end, :] .= lambda_tchaos(p1[end, :], 1.0, uq)

    for k = 1:3
        sol[i, j, k, 1] = mean(p1[k, :], uq.op)
        sol[i, j, k, 2] = std(p1[k, :], uq.op)
    end
end

plot(ps.x, sol[:, 2, 1, 1], label = "mean", xlabel = "x", ylabel = "ρ")
plot!(ps.x, sol[:, 2, 1, 1] .+ sol[:, 2, 1, 2], label = "mean+std")
plot!(ps.x, sol[:, 2, 1, 1] .- sol[:, 2, 1, 2], label = "mean-std")

cd(@__DIR__)
savefig("rho.png")
