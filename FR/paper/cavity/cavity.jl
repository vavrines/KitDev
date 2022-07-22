using OrdinaryDiffEq, BSON, KitBase, KitBase.Plots, KitBase.ProgressMeter
import FluxRC

function fboundary!(
    fh::T1,
    fb::T1,
    bc::T2,
    h::T3,
    b::T3,
    u::T4,
    v::T4,
    ω::T4,
    inK::Real,
    len::Real,
    rot::Real,
) where {
    T1<:AbstractArray{<:AbstractFloat,2},
    T2<:AbstractArray{<:Real,1},
    T3<:AbstractArray{<:AbstractFloat,2},
    T4<:AbstractArray{<:AbstractFloat,2},
}
    δ = heaviside.(u .* rot)

    SF = sum(ω .* u .* h .* (1.0 .- δ))
    SG =
        (bc[end] / π) *
        sum(ω .* u .* exp.(-bc[end] .* ((u .- bc[2]) .^ 2 .+ (v .- bc[3]) .^ 2)) .* δ)
    prim = [-SF / SG; bc[2:end]]

    MH = maxwellian(u, v, prim)
    MB = @. MH * inK / (2.0 * prim[end])

    fhWall = MH .* δ .+ h .* (1.0 .- δ)
    fbWall = MB .* δ .+ b .* (1.0 .- δ)

    @. fh = u * fhWall / (0.5 * len)
    @. fb = u * fbWall / (0.5 * len)

    return nothing
end

begin
    x0 = 0
    x1 = 1
    nx = 15
    y0 = 0
    y1 = 1
    ny = 15
    nxface = nx + 1
    nyface = ny + 1
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    deg = 2
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 28
    v0 = -5
    v1 = 5
    nv = 28
    inK = 1
    γ = 5 / 3
    knudsen = 0.075
    muref = ref_vhs_vis(knudsen, 1.0, 0.5)
    cfl = 0.15
    dt = cfl * dx / (u1 + 2.0)
    t = 0.0
    tmax = 20.0
    tspan = (0.0, tmax)
    nt = tmax ÷ dt |> Int
end

pspace = FluxRC.FRPSpace2D(x0, x1, nx, y0, y1, ny, deg)
vspace = VSpace2D(u0, u1, nu, v0, v1, nv)
δu = heaviside.(vspace.u)
δv = heaviside.(vspace.v)

begin
    xGauss = FluxRC.legendre_point(deg)
    ll = FluxRC.lagrange_point(xGauss, -1.0)
    lr = FluxRC.lagrange_point(xGauss, 1.0)
    lpdm = FluxRC.∂lagrange(xGauss)
    dgl, dgr = FluxRC.∂radau(deg, xGauss)
end

w0 = zeros(nx, ny, 4, nsp, nsp)
h0 = zeros(nx, ny, nu, nv, nsp, nsp)
b0 = zeros(nx, ny, nu, nv, nsp, nsp)
for i = 1:nx, j = 1:ny, p = 1:nsp, q = 1:nsp
    _prim = [1.0, 0.0, 0.0, 1.0]
    w0[i, j, :, p, q] .= prim_conserve(_prim, γ)
    h0[i, j, :, :, p, q] .= maxwellian(vspace.u, vspace.v, _prim)
    @. b0[i, j, :, :, p, q] = h0[i, j, :, :, p, q] * inK / (2.0 * _prim[end])
end
u0 = cat(h0, b0; dims = 7)
τ0 = zeros(nx, ny, nsp, nsp)

function mol!(du, u, p, t) # method of lines
    dx,
    dy,
    uvelo,
    vvelo,
    weights,
    δu,
    δv,
    MH,
    MB,
    fhx,
    fhy,
    fbx,
    fby,
    hx_face,
    hy_face,
    bx_face,
    by_face,
    fhx_face,
    fhy_face,
    fbx_face,
    fby_face,
    fhx_interaction,
    fhy_interaction,
    fbx_interaction,
    fby_interaction,
    rhs_h1,
    rhs_h2,
    rhs_b1,
    rhs_b2,
    inK,
    γ,
    muref,
    τ,
    ll,
    lr,
    lpdm,
    dgl,
    dgr = p

    dh = @view du[:, :, :, :, :, :, 1]
    db = @view du[:, :, :, :, :, :, 2]
    h = @view u[:, :, :, :, :, :, 1]
    b = @view u[:, :, :, :, :, :, 2]

    nx = size(h, 1)
    ny = size(h, 2)
    nu = size(h, 3)
    nv = size(h, 4)
    nsp = size(h, 5)

    #MH = similar(h); MB = similar(b)
    @inbounds Threads.@threads for q = 1:nsp
        for p = 1:nsp, j = 1:ny, i = 1:nx
            w = moments_conserve(
                h[i, j, :, :, p, q],
                b[i, j, :, :, p, q],
                uvelo,
                vvelo,
                weights,
            )
            prim = conserve_prim(w, γ)
            MH[i, j, :, :, p, q] .= maxwellian(uvelo, vvelo, prim)
            @. MB[i, j, :, :, p, q] = MH[i, j, :, :, p, q] * inK / (2.0 * prim[end])
            τ[i, j, p, q] = vhs_collision_time(prim, muref, 0.72)
        end
    end

    #fhx = similar(h); fhy = similar(h); fbx = similar(b); fby = similar(b)
    @inbounds Threads.@threads for q = 1:nsp
        for p = 1:nsp, l = 1:nv, k = 1:nu, j = 1:ny, i = 1:nx
            Jx = 0.5 * dx[i, j]
            fhx[i, j, k, l, p, q] = uvelo[k, l] * h[i, j, k, l, p, q] / Jx
            fbx[i, j, k, l, p, q] = uvelo[k, l] * b[i, j, k, l, p, q] / Jx
        end
    end
    @inbounds Threads.@threads for q = 1:nsp
        for p = 1:nsp, l = 1:nv, k = 1:nu, j = 1:ny, i = 1:nx
            Jy = 0.5 * dy[i, j]
            fhy[i, j, k, l, p, q] = vvelo[k, l] * h[i, j, k, l, p, q] / Jy
            fby[i, j, k, l, p, q] = vvelo[k, l] * b[i, j, k, l, p, q] / Jy
        end
    end

    #=hx_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)
    hy_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)
    bx_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)
    by_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)
    fhx_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)
    fbx_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)
    fhy_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)
    fby_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)=#
    hx_face .= 0.0
    hy_face .= 0.0
    bx_face .= 0.0
    by_face .= 0.0
    fhx_face .= 0.0
    fbx_face .= 0.0
    fhy_face .= 0.0
    fby_face .= 0.0
    @inbounds Threads.@threads for q = 1:nsp
        for l = 1:nv, k = 1:nu, j = 1:ny, i = 1:nx, p = 1:nsp
            hx_face[i, j, k, l, q, 1] += h[i, j, k, l, p, q] * lr[p]
            hx_face[i, j, k, l, q, 2] += h[i, j, k, l, p, q] * ll[p]
            bx_face[i, j, k, l, q, 1] += b[i, j, k, l, p, q] * lr[p]
            bx_face[i, j, k, l, q, 2] += b[i, j, k, l, p, q] * ll[p]

            fhx_face[i, j, k, l, q, 1] += fhx[i, j, k, l, p, q] * lr[p]
            fhx_face[i, j, k, l, q, 2] += fhx[i, j, k, l, p, q] * ll[p]
            fbx_face[i, j, k, l, q, 1] += fbx[i, j, k, l, p, q] * lr[p]
            fbx_face[i, j, k, l, q, 2] += fbx[i, j, k, l, p, q] * ll[p]
        end
    end
    @inbounds Threads.@threads for p = 1:nsp
        for l = 1:nv, k = 1:nu, j = 1:ny, i = 1:nx, q = 1:nsp
            hy_face[i, j, k, l, p, 1] += h[i, j, k, l, p, q] * lr[q]
            hy_face[i, j, k, l, p, 2] += h[i, j, k, l, p, q] * ll[q]
            by_face[i, j, k, l, p, 1] += b[i, j, k, l, p, q] * lr[q]
            by_face[i, j, k, l, p, 2] += b[i, j, k, l, p, q] * ll[q]

            fhy_face[i, j, k, l, p, 1] += fhy[i, j, k, l, p, q] * lr[q]
            fhy_face[i, j, k, l, p, 2] += fhy[i, j, k, l, p, q] * ll[q]
            fby_face[i, j, k, l, p, 1] += fby[i, j, k, l, p, q] * lr[q]
            fby_face[i, j, k, l, p, 2] += fby[i, j, k, l, p, q] * ll[q]
        end
    end

    #fhx_interaction = similar(u, nx+1, ny, nu, nv, nsp)
    #fbx_interaction = similar(u, nx+1, ny, nu, nv, nsp)
    #fhy_interaction = similar(u, nx, ny+1, nu, nv, nsp)
    #fby_interaction = similar(u, nx, ny+1, nu, nv, nsp)
    @inbounds for i = 2:nx, j = 1:ny, k = 1:nsp
        @. fhx_interaction[i, j, :, :, k] =
            fhx_face[i-1, j, :, :, k, 1] * δu + fhx_face[i, j, :, :, k, 2] * (1.0 - δu)
        @. fbx_interaction[i, j, :, :, k] =
            fbx_face[i-1, j, :, :, k, 1] * δu + fbx_face[i, j, :, :, k, 2] * (1.0 - δu)
    end
    @inbounds for i = 1:nx, j = 2:ny, k = 1:nsp
        @. fhy_interaction[i, j, :, :, k] =
            fhy_face[i, j-1, :, :, k, 1] * δv + fhy_face[i, j, :, :, k, 2] * (1.0 - δv)
        @. fby_interaction[i, j, :, :, k] =
            fby_face[i, j-1, :, :, k, 1] * δv + fby_face[i, j, :, :, k, 2] * (1.0 - δv)
    end

    # boundary
    @inbounds for i = 1:nsp
        for j = 1:ny
            fhwL = @view fhx_interaction[1, j, :, :, i]
            fbwL = @view fbx_interaction[1, j, :, :, i]
            fboundary!(
                fhwL,
                fbwL,
                [1.0, 0.0, 0.0, 1.0],
                hx_face[1, j, :, :, i, 2],
                bx_face[1, j, :, :, i, 2],
                uvelo,
                vvelo,
                weights,
                inK,
                dx[1, j],
                1.0,
            )

            fhwR = @view fhx_interaction[end, j, :, :, i]
            fbwR = @view fbx_interaction[end, j, :, :, i]
            fboundary!(
                fhwR,
                fbwR,
                [1.0, 0.0, 0.0, 1.0],
                hx_face[end, j, :, :, i, 1],
                bx_face[end, j, :, :, i, 1],
                uvelo,
                vvelo,
                weights,
                inK,
                dx[end, j],
                -1.0,
            )
        end
    end
    @inbounds for j = 1:nsp
        for i = 1:nx
            fhwD = @view fhy_interaction[i, 1, :, :, j]
            fbwD = @view fby_interaction[i, 1, :, :, j]
            fboundary!(
                fhwD,
                fbwD,
                [1.0, 0.0, 0.0, 1.0],
                hy_face[i, 1, :, :, j, 2],
                by_face[i, 1, :, :, j, 2],
                vvelo,
                -uvelo,
                weights,
                inK,
                dy[i, 1],
                1.0,
            )

            fhwU = @view fhy_interaction[i, end, :, :, j]
            fbwU = @view fby_interaction[i, end, :, :, j]
            fboundary!(
                fhwU,
                fbwU,
                [1.0, 0.0, -0.15, 1.0],
                hy_face[i, end, :, :, j, 1],
                by_face[i, end, :, :, j, 1],
                vvelo,
                -uvelo,
                weights,
                inK,
                dy[i, end],
                -1.0,
            )
        end
    end

    #rhs_h1 = zeros(eltype(u), nx, ny, nu, nv, nsp, nsp)
    #rhs_b1 = zeros(eltype(u), nx, ny, nu, nv, nsp, nsp)
    rhs_h1 .= 0.0
    rhs_b1 .= 0.0
    @inbounds for i = 1:nx, j = 1:ny, k = 1:nu, l = 1:nv, q = 1:nsp, p = 1:nsp, p1 = 1:nsp
        rhs_h1[i, j, k, l, p, q] += fhx[i, j, k, l, p1, q] * lpdm[p, p1]
        rhs_b1[i, j, k, l, p, q] += fbx[i, j, k, l, p1, q] * lpdm[p, p1]
    end

    #rhs_h2 = zeros(eltype(u), nx, ny, nu, nv, nsp, nsp)
    #rhs_b2 = zeros(eltype(u), nx, ny, nu, nv, nsp, nsp)
    rhs_h2 .= 0.0
    rhs_b2 .= 0.0
    @inbounds for i = 1:nx, j = 1:ny, k = 1:nu, l = 1:nv, p = 1:nsp, q = 1:nsp, q1 = 1:nsp
        rhs_h2[i, j, k, l, p, q] += fhy[i, j, k, l, p, q1] * lpdm[q, q1]
        rhs_b2[i, j, k, l, p, q] += fby[i, j, k, l, p, q1] * lpdm[q, q1]
    end

    @inbounds for i = 1:nx, j = 1:ny, k = 1:nu, l = 1:nv, p = 1:nsp, q = 1:nsp
        dh[i, j, k, l, p, q] =
            -(
                rhs_h1[i, j, k, l, p, q] +
                rhs_h2[i, j, k, l, p, q] +
                (fhx_interaction[i, j, k, l, q] - fhx_face[i, j, k, l, q, 2]) * dgl[p] +
                (fhx_interaction[i+1, j, k, l, q] - fhx_face[i, j, k, l, q, 1]) * dgr[p] +
                (fhy_interaction[i, j, k, l, p] - fhy_face[i, j, k, l, p, 2]) * dgl[q] +
                (fhy_interaction[i, j+1, k, l, p] - fhy_face[i, j, k, l, p, 1]) * dgr[q]
            ) + (MH[i, j, k, l, p, q] - h[i, j, k, l, p, q]) / τ[i, j, p, q]
        db[i, j, k, l, p, q] =
            -(
                rhs_b1[i, j, k, l, p, q] +
                rhs_b2[i, j, k, l, p, q] +
                (fbx_interaction[i, j, k, l, q] - fbx_face[i, j, k, l, q, 2]) * dgl[p] +
                (fbx_interaction[i+1, j, k, l, q] - fbx_face[i, j, k, l, q, 1]) * dgr[p] +
                (fby_interaction[i, j, k, l, p] - fby_face[i, j, k, l, p, 2]) * dgl[q] +
                (fby_interaction[i, j+1, k, l, p] - fby_face[i, j, k, l, p, 1]) * dgr[q]
            ) + (MB[i, j, k, l, p, q] - b[i, j, k, l, p, q]) / τ[i, j, p, q]
    end
end

begin
    MH = similar(h0)
    MB = similar(b0)
    fhx = similar(h0)
    fhy = similar(h0)
    fbx = similar(b0)
    fby = similar(b0)
    hx_face = zeros(nx, ny, nu, nv, nsp, 2)
    hy_face = zeros(nx, ny, nu, nv, nsp, 2)
    bx_face = zeros(nx, ny, nu, nv, nsp, 2)
    by_face = zeros(nx, ny, nu, nv, nsp, 2)
    fhx_face = zeros(nx, ny, nu, nv, nsp, 2)
    fbx_face = zeros(nx, ny, nu, nv, nsp, 2)
    fhy_face = zeros(nx, ny, nu, nv, nsp, 2)
    fby_face = zeros(nx, ny, nu, nv, nsp, 2)
    fhx_interaction = zeros(nx + 1, ny, nu, nv, nsp)
    fhy_interaction = zeros(nx, ny + 1, nu, nv, nsp)
    fbx_interaction = zeros(nx + 1, ny, nu, nv, nsp)
    fby_interaction = zeros(nx, ny + 1, nu, nv, nsp)
    rhs_h1 = zeros(nx, ny, nu, nv, nsp, nsp)
    rhs_b1 = zeros(nx, ny, nu, nv, nsp, nsp)
    rhs_h2 = zeros(nx, ny, nu, nv, nsp, nsp)
    rhs_b2 = zeros(nx, ny, nu, nv, nsp, nsp)
end

p = (
    pspace.dx,
    pspace.dy,
    vspace.u,
    vspace.v,
    vspace.weights,
    δu,
    δv,
    MH,
    MB,
    fhx,
    fhy,
    fbx,
    fby,
    hx_face,
    hy_face,
    bx_face,
    by_face,
    fhx_face,
    fhy_face,
    fbx_face,
    fby_face,
    fhx_interaction,
    fhy_interaction,
    fbx_interaction,
    fby_interaction,
    rhs_h1,
    rhs_h2,
    rhs_b1,
    rhs_b2,
    inK,
    γ,
    muref,
    τ0,
    ll,
    lr,
    lpdm,
    dgl,
    dgr,
)

prob = ODEProblem(mol!, u0, tspan, p)
itg = init(
    prob,
    Euler(),
    #ABDF2(),
    #TRBDF2(),
    #KenCarp3(),
    #KenCarp4(),
    #reltol = 1e-8,
    #abstol = 1e-8,
    save_everystep = false,
    adaptive = false,
    dt = dt,
    #autodiff = false,
)

u = itg.u
BSON.@save "sol.bson" u
@showprogress for iter = 1:nt
    step!(itg)
    if iter % 500 == 0
        u = itg.u
        BSON.@save "sol.bson" u
    end
end
#=
begin
    coord = zeros(nx * nsp, ny * nsp, 2)
    _prim = zeros(nx * nsp, ny * nsp, 4)
    for i = 1:nx, j = 1:ny
        idx0 = (i - 1) * nsp
        idy0 = (j - 1) * nsp

        for k = 1:nsp, l = 1:nsp
            idx = idx0 + k
            idy = idy0 + l
            coord[idx, idy, 1] = pspace.xp[i, j, k, l]
            coord[idx, idy, 2] = pspace.yp[i, j, k, l]

            _h = itg.u[i, j, :, :, k, l, 1]
            _b = itg.u[i, j, :, :, k, l, 2]
            _w = moments_conserve(_h, _b, vspace.u, vspace.v, vspace.weights)
            _prim[idx, idy, :] .= conserve_prim(_w, γ)
            _prim[idx, idy, 4] = 1 / _prim[idx, idy, 4]
        end

    end

    contourf(coord[1:end, 1, 1], coord[1, 1:end, 2], _prim[:, :, 2]')
end=#
