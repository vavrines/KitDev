using KitBase, FluxReconstruction, OrdinaryDiffEq, Langevin, LinearAlgebra, Plots
using ProgressMeter: @showprogress

cd(@__DIR__)

begin
    x0 = -1
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
    #u[i, j, 1] = 1.0
    #u[i, j, :] .+= uq.pce * 0.1 * sin(2π * ps.xpg[i, j])
    u[i, j, :] .= uq.pce * sin(π * ps.xpg[i, j])
end

a = zeros(uq.nm+1)
a[1] = 1.0

function dudt!(du, u, p, t)
    du .= 0.0
    J, ll, lr, lpdm, dgl, dgr, Δx, γ, uq, a = p

    nm = uq.nm
    nq = uq.nq

    ncell = size(u, 1)
    nsp = size(u, 2)

    u_ran = zeros(ncell, nsp, nq)
    for i = 1:ncell, j = 1:nsp
        u_ran[i, j, :] .= chaos_ran(u[i, j, :], uq)
    end
    a_ran = chaos_ran(a, uq)

    f = zeros(ncell, nsp, nm+1)
    for i = 1:ncell, j = 1:nsp
        _f = zeros(nq)
        for k = 1:nq
            _f[k] = u_ran[i, j, k] * a_ran[k] / J[i]
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

tspan = (0.0, 2.0)
p = (ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, ps.dx, γ, uq, a)
prob = ODEProblem(dudt!, u, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

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
sol0 = deepcopy(sol)

@showprogress for iter = 1:nt
    step!(itg)
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

plot(x, sol[:, 1], label="Numerical", lw=2, xlabel="x", ylabel="u")
plot!(x, sol0[:, 1], label="Exact", lw=2, line=:dash)
savefig("wave_mean.pdf")

plot(x, sol[:, 2], label="Numerical", lw=2, xlabel="x", ylabel="u")
plot!(x, sol0[:, 2], label="Exact", lw=2, line=:dash)
savefig("wave_std.pdf")
