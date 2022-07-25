using OrdinaryDiffEq, Plots, JLD2, ProgressMeter, LinearAlgebra, PyCall
using KitBase, FluxReconstruction

begin
    x0 = -25
    x1 = 25
    nx = 32
    dx = (x1 - x0) / nx
    deg = 2 # polynomial degree
    nsp = deg + 1
    u0 = -10
    u1 = 10
    nu = 200
    cfl = 0.1
    t = 0.0
    tspan = (0.0, 2.7)
    dt = cfl * dx / u1
    nt = floor(tspan[2] / dt) |> Int
    mach = 3.0
    knudsen = 1.0
end

begin
    ps = FRPSpace1D(x0, x1, nx, deg)
    vs = VSpace1D(u0, u1, nu)
    δ = heaviside.(vs.u)
    gas = Gas(knudsen, mach, 1.0, 0.0, 3.0, 0.81, 1.0, 0.5)
    ib = ib_rh(gas.Ma, gas.γ, vs.u)
end

begin
    cd(@__DIR__)

    mastr = mach |> Int |> string
    filename = "ma" * mastr * "_ref.jld2"
    #=
        @load filename x_ref u_ref
        prim_ref = zeros(length(x_ref), 3)
        for i = 1:size(u_ref, 1)
            idx0 = (i - 1) * size(u_ref, 3)
            idx = idx0+1:idx0+size(u_ref, 3)

            for j = 1:size(u_ref, 3)
                idx = idx0 + j
                _w = moments_conserve(u_ref[i, :, j], vs.u, vs.weights)
                prim_ref[idx, :] .= conserve_prim(_w, 3.0)
            end
        end=#

    @load filename x_ref prim_ref

    plot(x_ref, prim_ref[:, 1], legend = :none)
end

u = zeros(nx, nu, nsp)
for i = 1:nx, ppp1 = 1:nsp
    if i <= nx ÷ 2
        _prim = ib[2]
    else
        _prim = ib[6]
    end

    u[i, :, ppp1] .= maxwellian(vs.u, _prim)
end

function dudt!(du, u, p, t) # method of lines
    M,
    f,
    u_face,
    f_face,
    u_interaction,
    f_interaction,
    rhs,
    τ,
    dx,
    velo,
    weights,
    δ,
    ll,
    lr,
    lpdm,
    dgl,
    dgr,
    gam,
    μᵣ,
    ω = p

    ncell = length(dx)
    nu = length(velo)
    nsp = length(ll)
    nface = ncell + 1

    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            w = [
                sum(@. weights * u[i, :, k]),
                sum(@. weights * velo * u[i, :, k]),
                0.5 * sum(@. weights * velo^2 * u[i, :, k]),
            ]

            prim = conserve_prim(w, gam)
            M[i, :, k] .= maxwellian(velo, prim)
            τ[i, k] = 1.0#vhs_collision_time(prim, μᵣ, ω)
        end
    end

    @inbounds Threads.@threads for i = 1:ncell
        J = 0.5 * dx[i]

        for k = 1:nsp
            @. f[i, :, k] = velo * u[i, :, k] / J
        end
    end


    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:nu
            # right face of element i
            u_face[i, j, 1] = dot(u[i, j, :], lr)
            f_face[i, j, 1] = dot(f[i, j, :], lr)

            # left face of element i
            u_face[i, j, 2] = dot(u[i, j, :], ll)
            f_face[i, j, 2] = dot(f[i, j, :], ll)
        end
    end


    @inbounds Threads.@threads for i = 2:nface-1
        @. u_interaction[i, 1:nu] =
            u_face[i, 1:nu, 2] * (1.0 - δ) + u_face[i-1, 1:nu, 1] * δ
        @. f_interaction[i, 1:nu] =
            f_face[i, 1:nu, 2] * (1.0 - δ) + f_face[i-1, 1:nu, 1] * δ
    end


    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:nu, ppp1 = 1:nsp
            rhs[i, j, ppp1] = dot(f[i, j, :], lpdm[ppp1, :])
        end
    end

    @inbounds Threads.@threads for i = 2:ncell-1
        for ppp1 = 1:nsp, j = 1:nu
            du[i, j, ppp1] =
                -(
                    rhs[i, j, ppp1] +
                    (f_interaction[i, j] - f_face[i, j, 2]) * dgl[ppp1] +
                    (f_interaction[i+1, j] - f_face[i, j, 1]) * dgr[ppp1]
                ) + (M[i, j, ppp1] - u[i, j, ppp1]) / τ[i, ppp1]
        end
    end
    du[1, :, :] .= 0.0
    du[ncell, :, :] .= 0.0
end

begin
    M = zeros(nx, nu, nsp)
    f = zero(u)
    u_face = zeros(eltype(u), nx, nu, 2)
    f_face = zeros(eltype(u), nx, nu, 2)
    u_interaction = zeros(eltype(u), nx + 1, nu)
    f_interaction = zeros(eltype(u), nx + 1, nu)
    rhs = zeros(eltype(u), nx, nu, nsp)
    tau = zeros(nx, nsp)
end

p = (
    M,
    f,
    u_face,
    f_face,
    u_interaction,
    f_interaction,
    rhs,
    tau,
    ps.dx,
    vs.u,
    vs.weights,
    δ,
    ps.ll,
    ps.lr,
    ps.dl,
    ps.dhl,
    ps.dhr,
    gas.γ,
    gas.μᵣ,
    gas.ω,
)

prob = ODEProblem(dudt!, u, tspan, p)
itg = init(
    prob,
    BS3(),
    #TRBDF2(),
    #KenCarp3(),
    #KenCarp4(),
    saveat = tspan[2],
    #reltol = 1e-8,
    #abstol = 1e-8,
    adaptive = false,
    dt = dt,
)

@showprogress for iter = 1:nt
    step!(itg)
end

begin
    x = zeros(nx * nsp)
    prim = zeros(nx * nsp, 3)
    for i = 1:nx
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = ps.xpg[i, j]

            _h = itg.u[i, 1:nu, j]

            _w = moments_conserve(_h, vs.u, vs.weights)
            prim[idx, :] .= conserve_prim(_w, 3.0)
        end
    end
    plot(x, prim[:, 1])
    plot!(x_ref, prim_ref[:, 1], line = :dash)
end

begin
    itp = pyimport("scipy.interpolate")

    entry = 3
    f_ref = itp.interp1d(x_ref, prim_ref[:, entry], kind = "cubic")
    L1_error(prim[:, entry], f_ref(x), dx) |> println
    L2_error(prim[:, entry], f_ref(x), dx) |> println
    L∞_error(prim[:, entry], f_ref(x), dx) |> println
end
