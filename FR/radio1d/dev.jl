using OrdinaryDiffEq, LinearAlgebra, Plots
using KitBase, FluxReconstruction
using ProgressMeter: @showprogress

set = Setup(
    matter = "radiation",
    case = "inflow",
    space = "1d1f1v",
    boundary = "maxwell",
    cfl = 0.2,
    maxTime = 0.1,
)

deg = 2
ps = FRPSpace1D(0, 1, 50, deg)
vs = VSpace1D(-5.0, 5.0, 28)

δ = heaviside.(vs.u)

nsp = deg + 1
w = zeros(ps.nx, 3, nsp)
h = zeros(ps.nx, vs.nu, nsp)
for i = 1:ps.nx, ppp1 = 1:nsp
    if i <= ps.nx÷2
        _ρ = 1.0
        _λ = 0.5
    else
        _ρ = 0.125
        _λ = 0.625
    end

    w[i, :, ppp1] .= prim_conserve([_ρ, 0.0, _λ], 3)
    h[i, :, ppp1] .= maxwellian(vs.u, [_ρ, 1.0, _λ])
end

function mol!(du, u, p, t) # method of lines
    dx, velo, weights, δ, ll, lr, lpdm, dgl, dgr = p

    ncell = length(dx)
    nu = length(velo)
    nsp = length(ll)

    h = @view u[:, 1:nu, :]
    b = @view u[:, nu+1:end, :]

    M = similar(u, ncell, nu, nsp)
    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            w = [
                sum(@. weights * u[i, 1:nu, k]),
                sum(@. weights * velo * u[i, 1:nu, k]),
                0.5 * (sum(@. weights * velo^2 * u[i, 1:nu, k]))
            ]

            prim = conserve_prim(w, 3)
            M[i, 1:nu, k] .= maxwellian(velo, prim)
        end
    end

    τ = 0.01

    f = zero(u)
    @inbounds Threads.@threads for i = 1:ncell
        J = 0.5 * dx[i]

        for k = 1:nsp
            @. f[i, 1:nu, k] = velo * h[i, :, k] / J
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
        @. u_interaction[i, 1:nu] = u_face[i, 1:nu, 2] * (1.0 - δ) + u_face[i-1, 1:nu, 1] * δ
        @. f_interaction[i, 1:nu] = f_face[i, 1:nu, 2] * (1.0 - δ) + f_face[i-1, 1:nu, 1] * δ
    end

    rhs = zeros(eltype(u), ncell, nu, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:nu, ppp1 = 1:nsp, k = 1:nsp
            rhs[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
        end
    end

    rhs = zeros(eltype(u), ncell, nu, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:nu, ppp1 = 1:nsp, k = 1:nsp
            rhs[i, j, ppp1] += u[i, j, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for i = 2:ncell-1
        for ppp1 = 1:nsp, j = 1:nu
            du[i, j, ppp1] =
                -(
                    rhs[i, j, ppp1] +
                    (f_interaction[i, j] - f_face[i, j, 2]) * dgl[ppp1] +
                    (f_interaction[i+1, j] - f_face[i, j, 1]) * dgr[ppp1]
                ) + (M[i, j, ppp1] - u[i, j, ppp1]) / τ
        end
    end
    du[1, :, :] .= 0.0
    du[ncell, :, :] .= 0.0
end

u0 = zeros(ps.nx, vs.nu, nsp)
for i in axes(u0, 1), k in axes(u0, 3)
    j = 1:vs.nu
    u0[i, j, k] .= h[i, :, k]
end

tspan = (0.0, set.maxTime)
dt = set.cfl * ps.dx[1] / (vs.u1 + 1.0)
nt = tspan[2] / dt |> floor |> Int
p = (ps.dx, vs.u, vs.weights, δ, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr)

prob = ODEProblem(mol!, u0, tspan, p)
# integrator
itg = init(
    prob,
    Euler(),
    #TRBDF2(),
    #KenCarp3(),
    #KenCarp4(),
    saveat = tspan[2],
    #reltol = 1e-8,
    #abstol = 1e-8,
    adaptive = false,
    dt = dt,
    #autodiff = false,
)

@showprogress for iter = 1:nt
    step!(itg)
end

u = deepcopy(u0)
du = zero(u)
@showprogress for iter = 1:nt
    mol!(du, u, p, t)
    u .+= du * dt
    u[1,:,:] .= u[2,:,:]
    u[nx,:,:] .= u[nx-1,:,:]
end

begin
    x = zeros(nx * nsp)
    prim = zeros(nx * nsp, 3)
    for i = 1:nx
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = xsp[i, j]

            _h = itg.u[i, 1:nu, j]
            _b = itg.u[i, nu+1:end, j]
            _w = moments_conserve(_h, _b, vspace.u, vspace.weights)
            prim[idx, :] .= conserve_prim(_w, 5/3)
        end
    end
    plot(x, prim[:, 1])
    plot!(x, 1 ./ prim[:, 3])
end
