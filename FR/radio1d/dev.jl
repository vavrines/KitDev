using OrdinaryDiffEq, LinearAlgebra, Plots
using KitBase, FluxReconstruction
using ProgressMeter: @showprogress

begin
    deg = 2
    nsp = deg + 1
    knudsen = 1.0
end

ℓ = FR.basis_norm(deg)

set = Setup(
    matter = "radiation",
    case = "inflow",
    space = "1d1f1v",
    boundary = "maxwell",
    cfl = 0.1,
    maxTime = 0.5,
)
ps = FRPSpace1D(0, 1, 50, deg)
vs = VSpace1D(-1, 1, 48)

δ = heaviside.(vs.u)
function fb!(ff, f, u, rot = 1)
    δ = heaviside.(u .* rot)
    fWall = @. 0.5 * δ + f * (1.0 - δ)
    #fWall = 0.1
    @. ff = u * fWall

    return nothing
end

function mol!(du, u, p, t)
    dx, velo, weights, δ, kn, ll, lr, lpdm, dgl, dgr = p

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
    fw = @view f_interaction[1, :]
    fb!(fw, u_face[1, :, 2], velo)

    rhs = zeros(eltype(u), ncell, nu, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:nu, ppp1 = 1:nsp, k = 1:nsp
            rhs[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for i = 1:ncell-1
        for ppp1 = 1:nsp
            M = sum(weights .* u[i, :, ppp1])
            τ = kn

            j = 1:nu
            du[i, j, ppp1] .=
                -(
                    rhs[i, j, ppp1] .+
                    (f_interaction[i, j] .- f_face[i, j, 2]) .* dgl[ppp1] .+
                    (f_interaction[i+1, j] .- f_face[i, j, 1]) .* dgr[ppp1]
                ) #.+ (M .- u[i, j, ppp1]) ./ τ
        end
    end

    du[ncell, :, :] .= 0.0
end

u0 = ones(Float64, ps.nx, vs.nu, nsp) .* 1e-3
tspan = (0.0, set.maxTime)
dt = set.cfl * minimum(ps.dx) / (vs.u1 + 2)
nt = floor(tspan[2] / dt) |> Int
p = (ps.dx, vs.u, vs.weights, δ, knudsen, 
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

@showprogress for iter = 1:200#nt
    step!(itg)

    for j = 1:vs.nu, i = 1:ps.nx
        ũ = ps.iV * itg.u[i, j, :]
        su = ũ[end]^2 / sum(ũ.^2)
        isShock = shock_detector(log10(su), ps.deg)

        if isShock
            modal_filter!(ũ, ℓ; filter = :lasso)
            itg.u[i, j, :] .= ps.V * ũ
        end

        tmp = @view itg.u[i, j, :]
        positive_limiter(tmp, ps.wp ./ 2, ps.ll, ps.lr)
    end
end

begin
    x = zeros(ps.nx * nsp)
    sol = zeros(ps.nx * nsp, 3)
    for i = 1:ps.nx
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = ps.xpg[i, j]

            sol[idx, 1] = sum(itg.u[i, :, j] .* vs.weights)
        end
    end
end

plot(x[1:end], sol[1:end, 1])
scatter(x[1:end], sol[1:end, 1])
