using KitBase, FluxReconstruction, OffsetArrays, FastGaussQuadrature, LinearAlgebra,
    OrdinaryDiffEq
using ProgressMeter: @showprogress

using Plots
pyplot()

cd(@__DIR__)

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
ps0 = KitBase.CSpace2D(1.0, 6.0, 60, 0.0, π, 50)

deg = set.interpOrder - 1

function FR.FRPSpace2D(
    base::KitBase.CSpace2D,
    deg::Integer
)
    r = legendre_point(deg) .|> eltype(base.x)
    J = rs_jacobi(r, base.vertices)
    
    xpg = OffsetArray{eltype(base.x)}(undef, axes(base.x, 1), axes(base.y, 2), 1:deg+1, 1:deg+1, 1:2)
    for i in axes(xpg, 1), j in axes(xpg, 2), k = 1:deg+1, l = 1:deg+1
        @. xpg[i, j, k, l, :] = 
            (r[k] - 1.0) * (r[l] - 1.0) / 4 * base.vertices[i, j, 1, :] +
            (r[k] + 1.0) * (1.0 - r[l]) / 4 * base.vertices[i, j, 2, :] +
            (r[k] + 1.0) * (r[l] + 1.0) / 4 * base.vertices[i, j, 3, :] +
            (1.0 - r[k]) * (r[l] + 1.0) / 4 * base.vertices[i, j, 4, :]
    end

    w = gausslegendre(deg + 1)[2] .|> eltype(base.x)
    wp = [w[i] * w[j] for i = 1:deg+1, j = 1:deg+1]

    ll = lagrange_point(r, -1.0)
    lr = lagrange_point(r, 1.0)
    lpdm = ∂lagrange(r)

    V = vandermonde_matrix(deg, r)
    dVf = ∂vandermonde_matrix(deg, [-1.0, 1.0])
    ∂lf = zeros(eltype(base.x), 2, deg + 1)
    for i = 1:2
        ∂lf[i, :] .= V' \ dVf[i, :]
    end
    dll = ∂lf[1, :]
    dlr = ∂lf[2, :]

    dhl, dhr = ∂radau(deg, r)

    return FRPSpace2D{typeof(base),typeof(deg),typeof(J),typeof(r),typeof(xpg),typeof(wp)}(
        base,
        deg,
        J,
        (deg + 1)^2,
        r,
        xpg,
        wp,
        lpdm,
        ll,
        lr,
        dll,
        dlr,
        dhl,
        dhr,
    )
end

ps = FRPSpace2D(ps0, deg)

vs = nothing
gas = Gas(
    1e-6,
    2.0, # Mach
    1.0,
    1.0, # K
    5/3,
    0.81,
    1.0,
    0.5,
)
ib = nothing

ks = SolverSet(set, ps0, vs, gas, ib)

u0 = zeros(ps.nr, ps.nθ, deg+1, deg+1, 4)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    prim = [1.0, ks.gas.Ma, 0.0, 1.0]
    u0[i, j, k, l, :] .= prim_conserve(prim, ks.gas.γ)
end

n1 = [[[0.0, 0.0], [0.0, 0.0]] for i = 1:ps.nr, j = 1:ps.nθ]
for i = 1:ps.nr, j = 1:ps.nθ
    angle = sum(ps.dθ[1, 1:j-1]) + 0.5 * ps.dθ[1, j]
    n1[i, j][1] .= [cos(angle), sin(angle)]
    n1[i, j][2] .= [cos(angle), sin(angle)]
end

n2 = [[[0.0, 0.0], [0.0, 0.0]] for i = 1:ps.nr, j = 1:ps.nθ]
for i = 1:ps.nr, j = 1:ps.nθ
    angle = π/2 + sum(ps.dθ[1, 1:j-1])
    n2[i, j][1] .= [cos(angle), sin(angle)]
    n2[i, j][2] .= [cos(angle + ps.dθ[1, j]), sin(angle + ps.dθ[1, j])]
end

function dudt!(du, u, p, t)
    J, ll, lr, dhl, dhr, lpdm, γ = p
    
    nx = size(u, 1)
    ny = size(u, 2)
    nsp = size(u, 3)

    f = zeros(nx, ny, nsp, nsp, 4, 2)
    for i in axes(f, 1), j in axes(f, 2), k = 1:nsp, l = 1:nsp
        fg, gg = euler_flux(u[i, j, k, l, :], γ)
        for m = 1:4
            f[i, j, k, l, m, :] .= inv(J[i, j][k, l]) * [fg[m], gg[m]]
        end
    end

    u_face = zeros(nx, ny, 4, nsp, 4)
    f_face = zeros(nx, ny, 4, nsp, 4, 2)
    for i in axes(u_face, 1), j in axes(u_face, 2), l = 1:nsp, m = 1:4
        u_face[i, j, 1, l, m] = dot(u[i, j, l, :, m], ll)
        u_face[i, j, 2, l, m] = dot(u[i, j, :, l, m], lr)
        u_face[i, j, 3, l, m] = dot(u[i, j, l, :, m], lr)
        u_face[i, j, 4, l, m] = dot(u[i, j, :, l, m], ll)

        for n = 1:2
            f_face[i, j, 1, l, m, n] = dot(f[i, j, l, :, m, n], ll)
            f_face[i, j, 2, l, m, n] = dot(f[i, j, :, l, m, n], lr)
            f_face[i, j, 3, l, m, n] = dot(f[i, j, l, :, m, n], lr)
            f_face[i, j, 4, l, m, n] = dot(f[i, j, :, l, m, n], ll)
        end
    end

    fx_interaction = zeros(nx+1, ny, nsp, 4)
    for i = 2:nx, j = 1:ny, k = 1:nsp
        fw = @view fx_interaction[i, j, k, :]
        uL = local_frame(u_face[i-1, j, 2, k, :], n1[i, j][1][1], n1[i, j][1][2])
        uR = local_frame(u_face[i, j, 4, k, :], n1[i, j][1][1], n1[i, j][1][2])
        flux_hll!(fw, uL, uR, γ, 1.0)
    end
    fy_interaction = zeros(nx, ny+1, nsp, 4)
    for i = 1:nx, j = 2:ny, k = 1:nsp
        fw = @view fy_interaction[i, j, k, :]
        uL = local_frame(u_face[i, j-1, 3, k, :], n2[i, j][1][1], n2[i, j][1][2])
        uR = local_frame(u_face[i, j, 1, k, :], n2[i, j][1][1], n2[i, j][1][2])

        flux_hll!(fw, uL, uR, γ, 1.0)
    end
    
    rhs1 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
        rhs1[i, j, k, l, m] = dot(f[i, j, :, l, m, 1], lpdm[k, :])
    end
    rhs2 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
        rhs2[i, j, k, l, m] = dot(f[i, j, k, :, m, 2], lpdm[l, :])
    end

    for i = 2:nx-1, j = 2:ny-1, k = 1:nsp, l = 1:nsp, m = 1:4
        du[i, j, k, l, m] =
            -(
                rhs1[i, j, k, l, m] + rhs2[i, j, k, l, m] #+
                #(fx_interaction[i, j, l, m] - f_face[i, j, 4, l, m, 1]) * dhl[k] +
                #(fx_interaction[i+1, j, l, m] - f_face[i, j, 2, l, m, 1]) * dhr[k] +
                #(fy_interaction[i, j, k, m] - f_face[i, j, 1, k, m, 2]) * dhl[l] +
                #(fy_interaction[i, j+1, k, m] - f_face[i, j, 3, k, m, 2]) * dhr[l]
            )
    end

    return nothing
end

tspan = (0.0, 1.0)
p = (ps.J, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, ks.gas.γ)
prob = ODEProblem(dudt!, u0, tspan, p)

dt = 0.001
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Euler(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:2#nt
    step!(itg)
end




Plots.contourf(ps.x, ps.y, itg.u[:, :, 2, 2, 1], aspect_ratio=1, legend=true)


du = zero(u0)
tmp = dudt!(du, u0, p, 0.0)


ps.vertices[1, 1, :, :]


rs_jacobi(ps.xpl)


ps.xpl