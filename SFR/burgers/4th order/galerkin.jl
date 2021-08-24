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
    parameter1 = 0.9
    parameter2 = 1.1
end

uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)

u = zeros(ncell, nsp, uq.nm+1)
for i = 1:ncell, j = 1:nsp
    u[i, j, :] .= uq.pce * sin(2π * ps.xpg[i, j])
end

function dudt!(du, u, p, t)
    du .= 0.0
    J, ll, lr, lpdm, dgl, dgr, Δx, γ, uq = p

    nm = uq.nm
    nq = uq.nq

    ncell = size(u, 1)
    nsp = size(u, 2)

    u_ran = zeros(ncell, nsp, nq)
    for i = 1:ncell, j = 1:nsp
        u_ran[i, j, :] .= chaos_ran(u[i, j, :], uq)
    end

    f = zeros(ncell, nsp, nm+1)
    for i = 1:ncell, j = 1:nsp
        _f = zeros(nq)
        for k = 1:nq
            _f[k] = 0.5 * u_ran[i, j, k]^2 / J[i]
        end

        f[i, j, :] .= ran_chaos(_f, uq)
    end

    u_face = zeros(ncell, nm+1, 2)
    f_face = zeros(ncell, nm+1, 2)
    for i = 1:ncell, k = 1:nm+1
        u_face[i, k, 1] = dot(u[i, :, k], lr)
        f_face[i, k, 1] = dot(f[i, :, k], lr)
        u_face[i, k, 2] = dot(u[i, :, k], ll)
        f_face[i, k, 2] = dot(f[i, :, k], ll)
    end

    f_interaction = zeros(ncell + 1, nm+1)
    for i = 2:ncell
        f_interaction[i, :] .= (f_face[i-1, :, 1] .+ f_face[i, :, 2]) ./ 2 - 
            (Δx[i-1] + Δx[i]) / 2 * (u_face[i, :, 2] .- u_face[i-1, :, 1])
    end
    f_interaction[1, :] .= (f_face[ncell, :, 1] .+ f_face[1, :, 2]) ./ 2 - 
        (Δx[ncell] + Δx[1]) / 2 * (u_face[1, :, 2] .- u_face[ncell, :, 1])
    f_interaction[ncell+1, :] .= (f_face[ncell, :, 1] .+ f_face[1, :, 2]) ./ 2 - 
        (Δx[ncell] + Δx[1]) / 2 * (u_face[1, :, 2] .- u_face[ncell, :, 1])

    rhs1 = zero(u)
    for i = 1:ncell, ppp1 = 1:nsp, l = 1:nm+1
        rhs1[i, ppp1, l] = dot(f[i, :, l], lpdm[ppp1, :])
    end

    idx = 1:ncell
    for i in idx, ppp1 = 1:nsp, l = 1:nm+1
        du[i, ppp1, l] =
            -(
                rhs1[i, ppp1, l] +
                (f_interaction[i, l] - f_face[i, l, 2]) * dgl[ppp1] +
                (f_interaction[i+1, l] - f_face[i, l, 1]) * dgr[ppp1]
            )
    end
end

tspan = (0.0, 0.4)
p = (ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, ps.dx, γ, uq)
prob = ODEProblem(dudt!, u, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)
end

sol = zeros(ncell, nsp, 2)
for i in axes(sol, 1), j in axes(sol, 2)
    sol[i, j, 1] = mean(itg.u[i, j, :], uq.op)
    sol[i, j, 2] = std(itg.u[i, j, :], uq.op)
end

plot(ps.x, sol[:, 2, 1], label="mean", xlabel="x", ylabel="u")
plot!(ps.x, sol[:, 2, 1] .+ sol[:, 2, 2], label="mean+std")
plot!(ps.x, sol[:, 2, 1] .- sol[:, 2, 2], label="mean-std")

cd(@__DIR__)
savefig("burgers.pdf")
