using KitBase, FluxRC, OrdinaryDiffEq

begin
    x0 = 0
    x1 = 1
    ncell = 100
    nface = ncell + 1
    dx = (x1 - x0) / ncell
    deg = 2 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.2
    dt = cfl * dx
    t = 0.0
end

ps = FRPSpace1D(x0, x1, ncell, deg)

begin
    xFace = collect(x0:dx:x1)
    xGauss = legendre_point(deg)
    xsp = global_sp(xFace, xGauss)
    ll = lagrange_point(xGauss, -1.0)
    lr = lagrange_point(xGauss, 1.0)
    lpdm = ∂lagrange(xGauss)
end

u = zeros(ncell, nsp, 3)
for i = 1:ncell, ppp1 = 1:nsp
    if ps.x[i] <= 0.5
        prim = [1.0, 0.0, 0.5]
    else
        prim = [0.125, 0.0, 0.625]
    end
    u[i, ppp1, :] .= prim_conserve(prim, γ)
end

function dudt!(du, u, p, t) # method of lines
    xFace, e2f, f2e, a, deg, ll, lr, lpdm = p

    ncell = size(u, 1)
    nsp = size(u, 2)
    
    f = zeros(ncell, nsp)
    for i in 1:ncell, j in 1:nsp
        J = (xFace[i+1] - xFace[i]) / 2.0
        f[i, j] = advection_flux(u[i, j], a) / J
    end

    u_face = zeros(ncell, nsp)
    f_face = zeros(ncell, nsp)
    for i in 1:ncell, j in 1:nsp
        # right face of element i
        u_face[i, 1] += u[i, j] * lr[j]
        f_face[i, 1] += f[i, j] * lr[j]

        # left face of element i
        u_face[i, 2] += u[i, j] * ll[j]
        f_face[i, 2] += f[i, j] * ll[j]
    end

    au = zeros(nface)
    for i = 1:nface
        au[i] =
            (f_face[f2e[i, 1], 2] - f_face[f2e[i, 2], 1]) /
            (u_face[f2e[i, 1], 2] - u_face[f2e[i, 2], 1] + 1e-6)
    end

    f_interaction = zeros(nface)
    for i = 1:nface
        f_interaction[i] = (
            0.5 * (f_face[f2e[i, 2], 1] + f_face[f2e[i, 1], 2]) -
            0.5 * abs(au[i]) * (u_face[f2e[i, 1], 2] - u_face[f2e[i, 2], 1])
        )
    end

    dgl, dgr = ∂radau(deg, xGauss)

    rhs1 = zeros(ncell, nsp)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:nsp
        rhs1[i, ppp1] += f[i, k] * lpdm[ppp1, k]
    end

    for i in 1:ncell, ppp1 in 1:nsp
        du[i, ppp1] =
            -(
                rhs1[i, ppp1] +
                (f_interaction[e2f[i, 2]] - f_face[i, 2]) * dgl[ppp1] +
                (f_interaction[e2f[i, 1]] - f_face[i, 1]) * dgr[ppp1]
            )
    end
end

