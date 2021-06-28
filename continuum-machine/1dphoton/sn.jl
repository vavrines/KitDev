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
    set = Setup("radiation", "linesource", "1d1f1v", "kfvs", "bgk", 1, 2, "vanleer", "extra", 0.5, 0.3)

    # physical space
    x0 = 0
    x1 = 1
    nx = 100
    nxg = 0
    ps = PSpace1D(x0, x1, nx, nxg)

    # velocity space
    nu = 28
    points, weights = gausslegendre(nu)
    vs = VSpace1D(points[1], points[end], nu, points, ones(nu) .* (points[end] - points[1]) / (nu - 1), weights)

    # material
    σs = ones(Float32, nx)
    σa = zeros(Float32, nx)
    σt = σs + σa
    σq = zeros(Float32, nx)

    # time
    dt = set.cfl * ps.dx[1]
    nt = set.maxTime / dt |> floor |> Int
    global t = 0.0

    # solution
    f0 = 0.0001 * ones(nu)
    phi = zeros(nu, nx)
    for i = 1:nx
        phi[:, i] .= f0
    end

    flux = zeros(nu, nx+1)

    cd(@__DIR__)
end

for iter = 1:nt
    println("iteration $iter of $nt")

    ff = @view flux[:, 1]
    flux_wall!(ff, phi[:, 1], points, dt, 1.0)

    @inbounds for i = 2:nx
        ff = @view flux[:, i]
        KitBase.flux_kfvs!(ff, phi[:, i-1], phi[:, i], points, dt)
    end

    @inbounds for i = 1:nx-1
        for q = 1:nu
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (σs[i] * phi[q, i] - σt[i] * phi[q, i]) * dt +
                σq[i] * dt
        end
    end
    phi[:, nx] .=  phi[:, nx-1]

    global t += dt
end

ρ = [sum(phi[:, i] .* weights) for i = 1:nx]
plot(ps.x[1:nx], ρ)
