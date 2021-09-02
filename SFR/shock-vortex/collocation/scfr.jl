using KitBase, FluxReconstruction, Langevin, LinearAlgebra, OrdinaryDiffEq, OffsetArrays, Plots, JLD2
using ProgressMeter: @showprogress
using Base.Threads: @threads

begin
    set = Setup(
        "gas",
        "cylinder",
        "2d0f0v",
        "hll",
        "nothing",
        1, # species
        3, # order of accuracy
        "positivity", # limiter
        "euler",
        0.1, # cfl
        1.0, # time
    )
    ps = FRPSpace2D(0.0, 2.0, 100, 0.0, 1.0, 50, set.interpOrder-1, 1, 1)
    vs = nothing
    gas = Gas(
        1e-6,
        1.12, # Mach
        1.0,
        3.0, # K
        7/5,
        0.81,
        1.0,
        0.5,
    )
    ib = nothing
    ks = SolverSet(set, ps, vs, gas, ib)
end

begin
    uqMethod = "collocation"
    nr = 5
    nRec = 10
    opType = "uniform"
    parameter1 = 0.95
    parameter2 = 1.05

    uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)
end
l2 = [uq.t2Product[j-1, j-1] for j = 1:uq.nm+1]

cd(@__DIR__)
include("../ic.jl")
include("../../filter.jl")

function dudt!(du, u, p, t)
    du .= 0.0

    f, u_face, f_face, fx_interaction, fy_interaction, rhs1, rhs2,
    iJ, ll, lr, dhl, dhr, lpdm, γ, uq = p
    
    nx = size(u, 1) - 2
    ny = size(u, 2) - 2
    nsp = size(u, 3)
    nq = size(u, 6)

    @inbounds @threads for l = 1:nsp
        for k = 1:nsp, j in axes(f, 2), i in axes(f, 1)
            for n = 1:nq
                fg, gg = euler_flux(u[i, j, k, l, :, n], γ)
                for s = 1:4
                    f[i, j, k, l, s, n, :] .= iJ[i, j][k, l] * [fg[s], gg[s]]
                end
            end
        end
    end

    @inbounds @threads for n = 1:nq
        for m = 1:4, l = 1:nsp, j in axes(u_face, 2), i in axes(u_face, 1)
            u_face[i, j, 1, l, m, n] = dot(u[i, j, l, :, m, n], ll)
            u_face[i, j, 2, l, m, n] = dot(u[i, j, :, l, m, n], lr)
            u_face[i, j, 3, l, m, n] = dot(u[i, j, l, :, m, n], lr)
            u_face[i, j, 4, l, m, n] = dot(u[i, j, :, l, m, n], ll)
        end
    end

    @inbounds @threads for o = 1:2
        for n = 1:nq, m = 1:4, l = 1:nsp, j in axes(u_face, 2), i in axes(u_face, 1)
            f_face[i, j, 1, l, m, n, o] = dot(f[i, j, l, :, m, n, o], ll)
            f_face[i, j, 2, l, m, n, o] = dot(f[i, j, :, l, m, n, o], lr)
            f_face[i, j, 3, l, m, n, o] = dot(f[i, j, l, :, m, n, o], lr)
            f_face[i, j, 4, l, m, n, o] = dot(f[i, j, :, l, m, n, o], ll)
        end
    end

    @inbounds @threads for k = 1:nsp
        for j = 1:ny, i = 1:nx+1
            fw = zeros(4, nq)
            for l = 1:nq
                tmp = @view fw[:, l]
                uL = @view u_face[i-1, j, 2, k, :, l]
                uR = @view u_face[i, j, 4, k, :, l]
                flux_hll!(tmp, uL, uR, γ, 1.0)
            end
            fx_interaction[i, j, k, :, :] .= fw
        end
    end
    @inbounds @threads for k = 1:nsp
        for j = 1:ny+1, i = 1:nx
            fw = zeros(4, nq)
            for l = 1:nq
                tmp = @view fw[:, l]
                uL = local_frame(u_face[i, j-1, 3, k, :, l], 0.0, 1.0)
                uR = local_frame(u_face[i, j, 1, k, :, l], 0.0, 1.0)
                flux_hll!(tmp, uL, uR, γ, 1.0)
                tmp .= global_frame(tmp, 0.0, 1.0)
            end
            fy_interaction[i, j, k, :, :] .= fw
        end
    end

    @inbounds @threads for n = 1:nq
        for m = 1:4, l = 1:nsp, k = 1:nsp, j = 1:ny, i = 1:nx
            rhs1[i, j, k, l, m, n] = dot(f[i, j, :, l, m, n, 1], lpdm[k, :])
        end
    end
    @inbounds @threads for n = 1:nq
        for m = 1:4, l = 1:nsp, k = 1:nsp, j = 1:ny, i = 1:nx
            rhs2[i, j, k, l, m, n] = dot(f[i, j, k, :, m, n, 2], lpdm[l, :])
        end
    end

    @inbounds @threads for n = 1:nq
        for m = 1:4, l = 1:nsp, k = 1:nsp, j = 1:ny, i = 1:nx
            du[i, j, k, l, m, n] =
                -(
                    rhs1[i, j, k, l, m, n] + rhs2[i, j, k, l, m, n] +
                    (fx_interaction[i, j, l, m, n] * iJ[i, j][k, l][1, 1] - f_face[i, j, 4, l, m, n, 1]) * dhl[k] +
                    (fx_interaction[i+1, j, l, m, n] * iJ[i, j][k, l][1, 1] - f_face[i, j, 2, l, m, n, 1]) * dhr[k] +
                    (fy_interaction[i, j, k, m, n] * iJ[i, j][k, l][2, 2] - f_face[i, j, 1, k, m, n, 2]) * dhl[l] +
                    (fy_interaction[i, j+1, k, m, n] * iJ[i, j][k, l][2, 2] - f_face[i, j, 3, k, m, n, 2]) * dhr[l]
                )
        end
    end

    return nothing
end

begin
    f = OffsetArray{Float64}(undef, 0:ks.ps.nx+1, 0:ks.ps.ny+1, ks.ps.deg+1, ks.ps.deg+1, 4, uq.nq, 2)
    u_face = OffsetArray{Float64}(undef, 0:ks.ps.nx+1, 0:ks.ps.ny+1, 4, ks.ps.deg+1, 4, uq.nq)
    f_face = OffsetArray{Float64}(undef, 0:ks.ps.nx+1, 0:ks.ps.ny+1, 4, ks.ps.deg+1, 4, uq.nq, 2)
    fx_interaction = zeros(ks.ps.nx+1, ks.ps.ny, ks.ps.deg+1, 4, uq.nq)
    fy_interaction = zeros(ks.ps.nx, ks.ps.ny+1, ks.ps.deg+1, 4, uq.nq)
    rhs1 = zeros(ks.ps.nx, ks.ps.ny, ks.ps.deg+1, ks.ps.deg+1, 4, uq.nq)
    rhs2 = zeros(ks.ps.nx, ks.ps.ny, ks.ps.deg+1, ks.ps.deg+1, 4, uq.nq)
end

p = (f, u_face, f_face, fx_interaction, fy_interaction, rhs1, rhs2,
    ps.iJ, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, ks.gas.γ, uq)
tspan = (0.0, 1.0)
dt = 0.001
nt = tspan[2] ÷ dt |> Int

# initial condition
u0 = OffsetArray{Float64}(undef, 0:ps.nx+1, 0:ps.ny+1, ps.deg+1, ps.deg+1, 4, uq.nq)
begin
    gam = gas.γ
    MaL = gas.Ma
    MaR = sqrt((MaL^2 * (gam - 1.0) + 2.0) / (2.0 * gam * MaL^2 - (gam - 1.0)))
    ratioT =
        (1.0 + (gam - 1.0) / 2.0 * MaL^2) * (2.0 * gam / (gam - 1.0) * MaL^2 - 1.0) /
        (MaL^2 * (2.0 * gam / (gam - 1.0) + (gam - 1.0) / 2.0))
    t1 = [1.0, MaL * sqrt(gam / 2.0), 0.0, 1.0]
    t2 = [
        t1[1] * (gam + 1.0) * MaL^2 / ((gam - 1.0) * MaL^2 + 2.0),
        MaR * sqrt(gam / 2.0) * sqrt(ratioT),
        0.0,
        t1[end] / ratioT,
    ]

    for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
        primRan = zeros(4, uq.nq)
        Ma = uq.pceSample .* ks.gas.Ma

        for n = 1:uq.nq
            if ps.x[i, j] <= ps.x1 * 0.25
                primRan[:, n] .= [t2[1], t1[2] - t2[2], 0.0, t2[end]]
            else
                primRan[:, n] .= [t1[1], 0.0, 0.0, t1[end]]

                tmp = @view primRan[:, n]
                vortex_ic!(tmp, ks.gas.γ, ps.xpg[i, j, k, l, 1], ps.xpg[i, j, k, l, 2])
            end
        end

        uRan = uq_prim_conserve(primRan, ks.gas.γ, uq)
        for m = 1:4
            u0[i, j, k, l, m, :] .= uRan[m, :]
        end
    end
end

prob = ODEProblem(dudt!, u0, tspan, p)
itg = init(prob, Midpoint(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)

    # filter
    @inbounds @threads for j in axes(itg.u, 2)
        for i in axes(itg.u, 1)
            #ũ = ps.iV * reshape(itg.u[i, j, 1:3, 1:3, 1, :], 9, :)
            #su = maximum([sum(ũ[3:end, j].^2) / sum(ũ[:, j].^2) for j = 1:uq.nq])
            #isShock = shock_detector(log10(su), ps.deg)

            if true#isShock
                #λ = sqrt(su) * 2 #2e-5
                λ1 = 5e-5
                #λ2 = 1e-12#5
                for s = 1:4, ss = 1:uq.nq
                    û = ps.iV * reshape(itg.u[i, j, 1:3, 1:3, s, ss], 9)
                    FR.modal_filter!(û, λ1; filter = :l2)
                    itg.u[i, j, :, :, s, ss] .= reshape(ps.V * û, 3, 3)
                end
            end
        end
    end

    # boundary
    itg.u[:, 0, :, :, :, :] .= itg.u[:, 1, :, :, :, :]
    itg.u[:, ps.ny+1, :, :, :, :] .= itg.u[:, ps.ny, :, :, :, :]
    itg.u[ps.nx+1, :, :, :, :, :] .= itg.u[ps.nx, :, :, :, :, :]

    if iter % 100 == 0
        t = round(itg.t, digits=3)
        filename = "iter_" * string(t) * ".jld2"
        u = itg.u
        @save filename u
    end
end

begin
    x = zeros(ps.nx*(ps.deg+1), ps.ny*(ps.deg+1))
    y = zeros(ps.nx*(ps.deg+1), ps.ny*(ps.deg+1))
    sol = zeros(ps.nx*(ps.deg+1), ps.ny*(ps.deg+1), 4, 2)

    for i = 1:ps.nx, j = 1:ps.ny
        idx0 = (i - 1) * (ps.deg+1)
        idy0 = (j - 1) * (ps.deg+1)

        for k = 1:ps.deg+1, l = 1:ps.deg+1
            idx = idx0 + k
            idy = idy0 + l
            x[idx, idy] = ps.xpg[i, j, k, l, 1]
            y[idx, idy] = ps.xpg[i, j, k, l, 2]

            primRan = uq_conserve_prim(itg.u[i, j, k, l, :, :], ks.gas.γ, uq)
            primRan[4, :] .= 1 ./ primRan[4, :]
            primChaos = zeros(4, uq.nm+1)
            for ii = 1:4
                primChaos[ii, :] .= ran_chaos(primRan[ii, :], uq)
            end

            for s = 1:4
                sol[idx, idy, s, 1] = mean(primChaos[s, :], uq.op)
                sol[idx, idy, s, 2] = std(primChaos[s, :], uq.op)
            end
        end
    end

    #contourf(x, y, sol[:, :, 1], aspect_ratio=1, legend=true)
    #plot!(x[:, 1], sol[:, end÷2+1, 1])
end

#contourf(x[:, 1], y[1, :], sol[:, :, 4, 1]', aspect_ratio=1, legend=true)
#plot(x[:, 1], sol[:, end÷2+1, 1, 1])

u = itg.u
@save "sol.jld2" x sol u
