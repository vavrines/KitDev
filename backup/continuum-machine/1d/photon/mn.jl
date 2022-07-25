using KitBase, LinearAlgebra
using KitBase.FastGaussQuadrature, KitBase.JLD2, KitBase.Plots
using ProgressMeter: @showprogress

function flux_wall!(
    ff::T1,
    f::T2,
    u::T3,
    dt,
    rot = 1,
) where {
    T1<:AbstractVector{<:AbstractFloat},
    T2<:AbstractVector{<:AbstractFloat},
    T3<:AbstractVector{<:AbstractFloat},
}
    δ = heaviside.(u .* rot)
    fWall = 0.5 .* δ .+ f .* (1.0 .- δ)
    @. ff = u * fWall * dt

    return nothing
end

begin
    # setup
    set = Setup(
        "radiation",
        "linesource",
        "1d1f1v",
        "kfvs",
        "bgk",
        1,
        2,
        "vanleer",
        "extra",
        0.5,
        0.3,
    )

    # physical space
    x0 = 0
    x1 = 1
    nx = 100
    nxg = 0
    ps = PSpace1D(x0, x1, nx, nxg)

    # velocity space
    nu = 100
    points, weights = gausslegendre(nu)
    vs = VSpace1D(
        points[1],
        points[end],
        nu,
        points,
        ones(nu) .* (points[end] - points[1]) / (nu - 1),
        weights,
    )

    # material
    σs = ones(Float32, nx)
    σa = zeros(Float32, nx)
    σt = σs + σa
    σq = zeros(Float32, nx)

    # moments
    L = 5
    ne = L + 1
    m = eval_sphermonomial(points, L)

    # time
    dt = set.cfl * ps.dx[1]
    nt = set.maxTime / dt |> floor |> Int
    global t = 0.0

    # solution
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)

    cd(@__DIR__)
end

begin
    # initial condition
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)
    flux = zeros(Float32, ne, nx + 1)
    fη = zeros(nu)
end

for iter = 1:nt
    println("iteration $iter of $nt")

    # mathematical optimizer
    @inbounds for i = 1:nx
        opt = KitBase.optimize_closure(
            α[:, i],
            m,
            weights,
            phi[:, i],
            KitBase.maxwell_boltzmann_dual,
        )
        α[:, i] .= opt.minimizer
        phi[:, i] .= KitBase.realizable_reconstruct(
            opt.minimizer,
            m,
            weights,
            KitBase.maxwell_boltzmann_dual_prime,
        )
    end

    flux_wall!(fη, maxwell_boltzmann_dual.(α[:, 1]' * m)[:], points, dt, 1.0)
    for k in axes(flux, 1)
        flux[k, 1] = sum(m[k, :] .* weights .* fη)
    end

    @inbounds for i = 2:nx
        KitBase.flux_kfvs!(
            fη,
            KitBase.maxwell_boltzmann_dual.(α[:, i-1]' * m)[:],
            KitBase.maxwell_boltzmann_dual.(α[:, i]' * m)[:],
            points,
            dt,
        )

        for k in axes(flux, 1)
            flux[k, i] = sum(m[k, :] .* weights .* fη)
        end
    end

    @inbounds for i = 1:nx-1
        for q = 1:1
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (σs[i] * phi[q, i] - σt[i] * phi[q, i]) * dt +
                σq[i] * dt
        end

        for q = 2:ne
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (-σt[i] * phi[q, i]) * dt
        end
    end
    phi[:, nx] .= phi[:, nx-1]

    global t += dt
end

plot(ps.x[1:nx], phi[1, :], label = "Mn")
plot!(ps.x[1:nx], ρ, label = "Sn")
