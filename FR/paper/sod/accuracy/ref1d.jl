using OrdinaryDiffEq, Plots, JLD2, ProgressMeter, LinearAlgebra, PyCall, Statistics
using KitBase, FluxReconstruction

begin
    x0 = 0
    x1 = 1
    nx = 500
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 2 # polynomial degree
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 200#400
    cfl = 0.1
    dt = cfl * dx / (u1 + 1.2)
    t = 0.0
    knudsen = 0.01#1.0
    mu = ref_vhs_vis(knudsen, 1.0, 0.5)
end

pspace = FRPSpace1D(x0, x1, nx, deg)
vspace = VSpace1D(u0, u1, nu)
δ = heaviside.(vspace.u)

begin
    xFace = collect(x0:dx:x1)
    xGauss = legendre_point(deg)
    xsp = global_sp(xFace, xGauss)
    ll = lagrange_point(xGauss, -1.0)
    lr = lagrange_point(xGauss, 1.0)
    lpdm = ∂lagrange(xGauss)
    dgl, dgr = ∂radau(deg, xGauss)
end

begin
    w0 = zeros(nx, 3, nsp)
    h0 = zeros(nx, nu, nsp)
    b0 = zeros(nx, nu, nsp)
    for i = 1:nx, ppp1 = 1:nsp
        if i <= nx ÷ 2
            _ρ = 1.0
            _λ = 0.5
        else
            _ρ = 0.125
            _λ = 0.625
        end

        w0[i, :, ppp1] .= prim_conserve([_ρ, 0.0, _λ], 5 / 3)
        h0[i, :, ppp1] .= maxwellian(vspace.u, [_ρ, 0.0, _λ])
        @. b0[i, :, ppp1] = h0[i, :, ppp1] * 2.0 / 2.0 / _λ
    end
end

function mol!(du, u, p, t) # method of lines
    dx, velo, weights, δ, knudsen, muref, ll, lr, lpdm, dgl, dgr = p

    ncell = length(dx)
    nu = length(velo)
    nsp = length(ll)
    n2 = size(u, 2)

    w = @view u[:, 1:3, :]
    h = @view u[:, 4:nu+3, :]
    b = @view u[:, nu+4:end, :]

    f = zero(u)
    @inbounds Threads.@threads for i = 1:ncell
        J = 0.5 * dx[i]

        for k = 1:nsp
            @. f[i, 4:nu+3, k] = velo * h[i, :, k] / J
            @. f[i, nu+4:end, k] = velo * b[i, :, k] / J

            f[i, 1, k] = sum(weights .* f[i, 4:nu+3, k])
            f[i, 2, k] = sum(weights .* velo .* f[i, 4:nu+3, k])
            f[i, 3, k] =
                0.5 * (
                    sum(weights .* velo .^ 2 .* f[i, 4:nu+3, k]) +
                    sum(weights .* f[i, nu+4:end, k])
                )
        end
    end

    pressure = zeros(eltype(u), ncell, nsp)
    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            _prim = conserve_prim(u[i, 1:3, k], 5 / 3)
            pressure[i, k] = 0.5 * _prim[1] / _prim[end]
        end
    end

    u_face = zeros(eltype(u), ncell, n2, 2)
    f_face = zeros(eltype(u), ncell, n2, 2)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:n2, k = 1:nsp
            # right face of element i
            u_face[i, j, 1] += u[i, j, k] * lr[k]
            f_face[i, j, 1] += f[i, j, k] * lr[k]

            # left face of element i
            u_face[i, j, 2] += u[i, j, k] * ll[k]
            f_face[i, j, 2] += f[i, j, k] * ll[k]
        end
    end

    p_face = zeros(eltype(u), ncell, 2)
    @inbounds Threads.@threads for i = 1:ncell
        for k = 1:nsp
            p_face[i, 1] += pressure[i, k] * lr[k]
            p_face[i, 2] += pressure[i, k] * ll[k]
        end
    end

    u_interaction = zeros(eltype(u), nface, n2)
    f_interaction = zeros(eltype(u), nface, n2)
    p_interaction = zeros(eltype(u), nface)
    @inbounds Threads.@threads for i = 2:nface-1
        @. u_interaction[i, 4:nu+3] =
            u_face[i, 4:nu+3, 2] * (1.0 - δ) + u_face[i-1, 4:nu+3, 1] * δ
        @. u_interaction[i, nu+4:end] =
            u_face[i, nu+4:end, 2] * (1.0 - δ) + u_face[i-1, nu+4:end, 1] * δ
        @. f_interaction[i, 4:nu+3] =
            f_face[i, 4:nu+3, 2] * (1.0 - δ) + f_face[i-1, 4:nu+3, 1] * δ
        @. f_interaction[i, nu+4:end] =
            f_face[i, nu+4:end, 2] * (1.0 - δ) + f_face[i-1, nu+4:end, 1] * δ

        u_interaction[i, 1] = sum(weights .* u_interaction[i, 4:nu+3])
        u_interaction[i, 2] = sum(weights .* velo .* u_interaction[i, 4:nu+3])
        u_interaction[i, 3] =
            0.5 * (
                sum(weights .* velo .^ 2 .* u_interaction[i, 4:nu+3]) +
                sum(weights .* u_interaction[i, nu+4:end])
            )
        f_interaction[i, 1] = sum(weights .* f_interaction[i, 4:nu+3])
        f_interaction[i, 2] = sum(weights .* velo .* f_interaction[i, 4:nu+3])
        f_interaction[i, 3] =
            0.5 * (
                sum(weights .* velo .^ 2 .* f_interaction[i, 4:nu+3]) +
                sum(weights .* f_interaction[i, nu+4:end])
            )

        _prim = conserve_prim(u_interaction[i, 1:3], 5 / 3)
        p_interaction[i] = 0.5 * _prim[1] / _prim[end]
    end

    ∇p = zeros(eltype(u), ncell, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for ppp1 = 1:nsp, k = 1:nsp
            ∇p[i, ppp1] += pressure[i, k] * lpdm[ppp1, k]
        end

        for ppp1 = 1:nsp
            ∇p[i, ppp1] +=
                (p_interaction[i] - p_face[i, 2]) * dgl[ppp1] +
                (p_interaction[i+1] - p_face[i, 1]) * dgr[ppp1]
        end
    end
    #=
        ∇u = zeros(eltype(u), ncell, n2, nsp)
        @inbounds Threads.@threads for i = 1:ncell
            for j = 1:n2, ppp1 = 1:nsp, k = 1:nsp
                ∇u[i, j, ppp1] += u[i, j, k] * lpdm[ppp1, k]
            end

            for j = 1:n2, ppp1 = 1:nsp
                ∇u[i, j, ppp1] += 
                    (u_interaction[i, j] - u_face[i, j, 2]) * dgl[ppp1] +
                    (u_interaction[i+1, j] - u_face[i, j, 1]) * dgr[ppp1]
            end
        end
    =#
    rhs = zeros(eltype(u), ncell, n2, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:n2, ppp1 = 1:nsp, k = 1:nsp
            rhs[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for i = 2:ncell-1
        for ppp1 = 1:nsp
            j = 1:3
            @. du[i, j, ppp1] = -(
                rhs[i, j, ppp1] +
                (f_interaction[i, j] - f_face[i, j, 2]) * dgl[ppp1] +
                (f_interaction[i+1, j] - f_face[i, j, 1]) * dgr[ppp1]
            )

            #fac = dx[i] / (knudsen / u[i, 1, ppp1] )
            #fac = 5.0 * dx[i] / knudsen
            fac = 0.0

            j = 4:nu+3
            du[i, j, ppp1] .=
                -(
                    rhs[i, j, ppp1] .+
                    (f_interaction[i, j] .- f_face[i, j, 2]) .* dgl[ppp1] .+
                    (f_interaction[i+1, j] .- f_face[i, j, 1]) .* dgr[ppp1]
                ) .+
                (
                    maxwellian(velo, conserve_prim(u[i, 1:3, ppp1], 5 / 3)) .-
                    u[i, j, ppp1]
                ) ./ (
                    vhs_collision_time(conserve_prim(u[i, 1:3, ppp1], 5 / 3), mu, 0.81) .+
                    fac * (abs(∇p[i, ppp1]) * dx[i] / mean(pressure[i, :]) * dt)
                )
            #(maxwellian(velo, conserve_prim(u[i, 1:3, ppp1], 5/3)) .- u[i, j, ppp1]) ./ (vhs_collision_time(conserve_prim(u[i, 1:3, ppp1], 5/3), mu, 0.81) .+ fac * (abs(∇p[i, ppp1]) * dx[i] / pressure[i, ppp1] * dt))
            #(maxwellian(velo, conserve_prim(u[i, 1:3, ppp1], 5/3)) .- u[i, j, ppp1]) ./ (vhs_collision_time(conserve_prim(u[i, 1:3, ppp1], 5/3), mu, 0.81) .+ fac * (abs(∇u[i, 2, ppp1] / u[i, 2, ppp1]) * dx[i] * dt))

            j = nu+4:n2
            du[i, j, ppp1] .=
                -(
                    rhs[i, j, ppp1] .+
                    (f_interaction[i, j] .- f_face[i, j, 2]) .* dgl[ppp1] .+
                    (f_interaction[i+1, j] .- f_face[i, j, 1]) .* dgr[ppp1]
                ) .+
                (
                    maxwellian(velo, conserve_prim(u[i, 1:3, ppp1], 5 / 3)) ./
                    conserve_prim(u[i, 1:3, ppp1], 5 / 3)[end] .- u[i, j, ppp1]
                ) ./ (
                    vhs_collision_time(conserve_prim(u[i, 1:3, ppp1], 5 / 3), mu, 0.81) .+
                    fac * (abs(∇p[i, ppp1]) * dx[i] / mean(pressure[i, :]) * dt)
                )
            #(maxwellian(velo, conserve_prim(u[i, 1:3, ppp1], 5/3)) ./ conserve_prim(u[i, 1:3, ppp1], 5/3)[end] .- u[i, j, ppp1]) ./ (vhs_collision_time(conserve_prim(u[i, 1:3, ppp1], 5/3), mu, 0.81) .+ fac * (abs(∇p[i, ppp1]) * dx[i] / pressure[i, ppp1] * dt))
            #(maxwellian(velo, conserve_prim(u[i, 1:3, ppp1], 5/3)) ./ conserve_prim(u[i, 1:3, ppp1], 5/3)[end] .- u[i, j, ppp1]) ./ (vhs_collision_time(conserve_prim(u[i, 1:3, ppp1], 5/3), mu, 0.81) .+ fac * (abs(∇u[i, 2, ppp1] / u[i, 2, ppp1]) * dx[i] * dt))
        end
    end

    #=@show vhs_collision_time(conserve_prim(u[25, 1:3, 1], 5/3), mu, 0.81)
    @show vhs_collision_time(conserve_prim(u[25, 1:3, 2], 5/3), mu, 0.81)
    @show vhs_collision_time(conserve_prim(u[25, 1:3, 3], 5/3), mu, 0.81)
    @show 500. * (abs(∇p[25, 1]) * dx[25] / mean(pressure[25, :]) * dt)
    @show 500. * (abs(∇p[25, 2]) * dx[25] / mean(pressure[25, :]) * dt)
    @show 500. * (abs(∇p[25, 3]) * dx[25] / mean(pressure[25, :]) * dt)
    @show vhs_collision_time(conserve_prim(u[26, 1:3, 1], 5/3), mu, 0.81)
    @show vhs_collision_time(conserve_prim(u[26, 1:3, 2], 5/3), mu, 0.81)
    @show vhs_collision_time(conserve_prim(u[26, 1:3, 3], 5/3), mu, 0.81)
    @show 500. * (abs(∇p[26, 1]) * dx[26] / mean(pressure[26, :]) * dt)
    @show 500. * (abs(∇p[26, 2]) * dx[26] / mean(pressure[26, :]) * dt)
    @show 500. * (abs(∇p[26, 3]) * dx[26] / mean(pressure[26, :]) * dt)=#

    #=@show vhs_collision_time(conserve_prim(u[25, 1:3, 3], 5/3), mu, 0.81)
    @show vhs_collision_time(conserve_prim(u[26, 1:3, 1], 5/3), mu, 0.81)
    @show abs(∇p[25, 3]) * dt * dx[25] / pressure[25, 3]
    @show abs(∇p[26, 1]) * dt * dx[26] / pressure[26, 1]=#

    du[1, :, :] .= 0.0
    du[ncell, :, :] .= 0.0
end

u0 = zeros(nx, 2 * nu + 3, nsp)
for i in axes(u0, 1), k in axes(u0, 3)
    u0[i, 1:3, k] .= w0[i, :, k]

    j = 4:nu+3
    u0[i, j, k] .= h0[i, :, k]

    j = nu+4:2*nu+3
    u0[i, j, k] .= b0[i, :, k]
end

tspan = (0.0, 0.12)
nt = floor(tspan[2] / dt) |> Int
p = (pspace.dx, vspace.u, vspace.weights, δ, knudsen, mu, ll, lr, lpdm, dgl, dgr)

prob = ODEProblem(mol!, u0, tspan, p)
itg = init(
    prob,
    BS3(),
    #ABDF2(),
    #TRBDF2(),
    #KenCarp3(),
    save_everystep = false,
    adaptive = false,
    dt = dt,
    progress = true,
    progress_steps = 10,
    progress_name = "frode",
)

@showprogress for iter = 1:nt
    step!(itg)
end

begin
    x = zeros(nx * nsp)
    w = zeros(nx * nsp, 3)
    prim = zeros(nx * nsp, 4)
    for i = 1:nx
        idx0 = (i - 1) * nsp
        for j = 1:nsp
            idx = idx0 + j
            x[idx] = xsp[i, j]

            w[idx, :] = itg.u[i, 1:3, j]
            prim[idx, 1:3] .= conserve_prim(w[idx, :], 5 / 3)
            prim[idx, 4] = 0.5 * prim[idx, 1] / prim[idx, 3]
        end
    end
    Plots.plot(x[1:end], prim[1:end, 1:2])
    Plots.plot!(x[1:end], prim[1:end, 4])
end

x_ref = x
u_ref = itg.u
@save "kn2_ref.jld2" x_ref u_ref
#@save "kn3_ref.jld2" x_ref u_ref
