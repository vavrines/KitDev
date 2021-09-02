using KitBase, FluxReconstruction, Langevin, LinearAlgebra, OrdinaryDiffEq, OffsetArrays, Plots, JLD2
using ProgressMeter: @showprogress
using Base.Threads: @threads

begin
    set = Setup(
        "gas",
        "cylinder",
        "2d0f",
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
    uqMethod = "galerkin"
    nr = 5
    nRec = 10
    opType = "uniform"
    parameter1 = 0.95
    parameter2 = 1.05

    uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)
end
l2 = [uq.t2Product[j-1, j-1] for j = 1:uq.nm+1]

cd(@__DIR__)
include("ic.jl")
include("../filter.jl")

function dudt!(du, u, p, t)
    du .= 0.0

    uRan, f, u_face, f_face, fx_interaction, fy_interaction, rhs1, rhs2,
    iJ, ll, lr, dhl, dhr, lpdm, γ, uq = p
    
    nx = size(u, 1) - 2
    ny = size(u, 2) - 2
    nsp = size(u, 3)
    nm = size(u, 6)
    nq = size(uRan, 6)

    @inbounds @threads for m = 1:4
        for l = 1:nsp, k = 1:nsp, j in axes(f, 2), i in axes(f, 1)
            uRan[i, j, k, l, m, :] .= chaos_ran(u[i, j, k, l, m, :], uq)
        end
    end

    @inbounds @threads for l = 1:nsp
        for k = 1:nsp, j in axes(f, 2), i in axes(f, 1)
            fRan = zeros(4, nq, 2)
            for n = 1:nq
                fg, gg = euler_flux(uRan[i, j, k, l, :, n], γ)
                for s = 1:4
                    fRan[s, n, :] .= iJ[i, j][k, l] * [fg[s], gg[s]]
                end
            end

            for d = 1:2, s = 1:4
                f[i, j, k, l, s, :, d] .= ran_chaos(fRan[s, :, d], uq)
            end
        end
    end

    @inbounds @threads for n = 1:nq
        for m = 1:4, l = 1:nsp, j in axes(u_face, 2), i in axes(u_face, 1)
            u_face[i, j, 1, l, m, n] = dot(uRan[i, j, l, :, m, n], ll)
            u_face[i, j, 2, l, m, n] = dot(uRan[i, j, :, l, m, n], lr)
            u_face[i, j, 3, l, m, n] = dot(uRan[i, j, l, :, m, n], lr)
            u_face[i, j, 4, l, m, n] = dot(uRan[i, j, :, l, m, n], ll)
        end
    end

    @inbounds @threads for o = 1:2
        for n = 1:nm, m = 1:4, l = 1:nsp, j in axes(u_face, 2), i in axes(u_face, 1)
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
            for s = 1:4
                fx_interaction[i, j, k, s, :] .= ran_chaos(fw[s, :], uq)
            end
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
            for s = 1:4
                fy_interaction[i, j, k, s, :] .= ran_chaos(fw[s, :], uq)
            end
        end
    end

    @inbounds @threads for n = 1:nm
        for m = 1:4, l = 1:nsp, k = 1:nsp, j = 1:ny, i = 1:nx
            rhs1[i, j, k, l, m, n] = dot(f[i, j, :, l, m, n, 1], lpdm[k, :])
        end
    end
    @inbounds @threads for n = 1:nm
        for m = 1:4, l = 1:nsp, k = 1:nsp, j = 1:ny, i = 1:nx
            rhs2[i, j, k, l, m, n] = dot(f[i, j, k, :, m, n, 2], lpdm[l, :])
        end
    end

    @inbounds @threads for n = 1:nm
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
    uRan = OffsetArray{Float64}(undef, 0:ps.nx+1, 0:ps.ny+1, ps.deg+1, ps.deg+1, 4, uq.nq)
    f = OffsetArray{Float64}(undef, 0:ks.ps.nx+1, 0:ks.ps.ny+1, ks.ps.deg+1, ks.ps.deg+1, 4, uq.nm+1, 2)
    u_face = OffsetArray{Float64}(undef, 0:ks.ps.nx+1, 0:ks.ps.ny+1, 4, ks.ps.deg+1, 4, uq.nq)
    f_face = OffsetArray{Float64}(undef, 0:ks.ps.nx+1, 0:ks.ps.ny+1, 4, ks.ps.deg+1, 4, uq.nm+1, 2)
    fx_interaction = zeros(ks.ps.nx+1, ks.ps.ny, ks.ps.deg+1, 4, uq.nm+1)
    fy_interaction = zeros(ks.ps.nx, ks.ps.ny+1, ks.ps.deg+1, 4, uq.nm+1)
    rhs1 = zeros(ks.ps.nx, ks.ps.ny, ks.ps.deg+1, ks.ps.deg+1, 4, uq.nm+1)
    rhs2 = zeros(ks.ps.nx, ks.ps.ny, ks.ps.deg+1, ks.ps.deg+1, 4, uq.nm+1)
end

p = (uRan, f, u_face, f_face, fx_interaction, fy_interaction, rhs1, rhs2,
    ps.iJ, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, ks.gas.γ, uq)
tspan = (0.0, 1.0)
dt = 0.001
nt = tspan[2] ÷ dt |> Int

# initial condition
u0 = OffsetArray{Float64}(undef, 0:ps.nx+1, 0:ps.ny+1, ps.deg+1, ps.deg+1, 4, uq.nm+1)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    primRan = zeros(4, uq.nq)
    Ma = uq.pceSample .* ks.gas.Ma

    for n = 1:uq.nq
        t1 = ib_rh(Ma[n], ks.gas.γ, rand(3))[2]
        t2 = ib_rh(Ma[n], ks.gas.γ, rand(3))[6]

        if ps.x[i, j] <= ps.x1 * 0.25
            primRan[:, n] .= [t2[1], t1[2] - t2[2], 0.0, t2[3]]
        else
            primRan[:, n] .= [t1[1], 0.0, 0.0, t1[3]]

            tmp = @view primRan[:, n]
            vortex_ic!(tmp, ks.gas.γ, ps.xpg[i, j, k, l, 1], ps.xpg[i, j, k, l, 2])
        end
    end

    uRan = uq_prim_conserve(primRan, ks.gas.γ, uq)

    for m = 1:4
        u0[i, j, k, l, m, :] .= ran_chaos(uRan[m, :], uq)
    end
end

prob = ODEProblem(dudt!, u0, tspan, p)
itg = init(prob, Midpoint(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)

    # filter
    for i in axes(itg.u, 1), j in axes(itg.u, 2)
        ũ = ps.iV * reshape(itg.u[i, j, 1:3, 1:3, 1, :], 9, :)
        su = maximum([sum(ũ[3:end, j].^2) / sum(ũ[:, j].^2) for j = 1:uq.nm+1])
        sv = maximum([ũ[j, end]^2 * uq.t2Product[uq.nm, uq.nm] / sum(ũ[j, :].^2 .* l2) for j = 1:ps.deg+1])
        isShock = max(shock_detector(log10(su), ps.deg), shock_detector(log10(sv), ps.deg))

        if isShock
            #λ = sqrt(su) * 2 #2e-5
            λ1 = 5e-5
            λ2 = 1e-12#5
            for s = 1:4
                û = ps.iV * reshape(itg.u[i, j, 1:3, 1:3, s, :], 9, :)
                
                FR.modal_filter!(û, λ1, λ2; filter = :l2)
                #FR.modal_filter!(û, ϕ; filter = :lasso)
                
                uNode = reshape(ps.V * û, 3, 3, :)
                itg.u[i, j, :, :, s, :] .= uNode
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

            uChaos = uq_conserve_prim(itg.u[i, j, k, l, :, :], ks.gas.γ, uq)
            for s = 1:4
                sol[idx, idy, s, 1] = mean(uChaos[s, :], uq.op)
                sol[idx, idy, s, 2] = std(uChaos[s, :], uq.op)
            end
        end
    end

    #contourf(x, y, sol[:, :, 1], aspect_ratio=1, legend=true)
    #plot!(x[:, 1], sol[:, end÷2+1, 1])
end

#contourf(x[:, 1], y[1, :], sol[:, :, 1, 1]', aspect_ratio=1, legend=true)
#plot(x[:, 1], sol[:, end÷2+1, 1, 1])

u = itg.u
@save "sol.jld2" x sol u
