using OrdinaryDiffEq, LinearAlgebra, Plots, PyCall, LinearAlgebra
using KitBase, FluxReconstruction
using Base.Threads: @threads
using ProgressMeter: @showprogress
import PyPlot

itp = pyimport("scipy.interpolate")
begin
    deg = 2
    nsp = deg + 1
    oq = 6
    knudsen = 1e0
    #ℓ = FR.basis_norm(deg)
end

set = Setup(
    matter = "radiation",
    case = "linesource",
    space = "2d1f2v",
    boundary = "fix",
    cfl = 0.2,
    maxTime = 0.1,
)
ps = FRPSpace2D(-3.5, 3.5, 20, -3.5, 3.5, 20, deg)

points, triangulation = octa_quadrature(oq)
weights = quadrature_weights(points, triangulation)
vs = VSpace1D(-1, 1, length(weights), points, zero(points), weights)
δu = heaviside.(vs.u[:, 1])
δv = heaviside.(vs.u[:, 2])

function is_absorb(x, y)
    cds = Array{Bool}(undef, 11) # conditions

    cds[1] = -2.5 < x < -1.5 && 1.5 < y < 2.5
    cds[2] = -2.5 < x < -1.5 && -0.5 < y < 0.5
    cds[3] = -2.5 < x < -1.5 && -2.5 < y < -1.5
    cds[4] = -1.5 < x < -0.5 && 0.5 < y < 1.5
    cds[5] = -1.5 < x < -0.5 && -1.5 < y < -0.5
    cds[6] = -0.5 < x < 0.5 && -2.5 < y < -1.5
    cds[7] = 0.5 < x < 1.5 && 0.5 < y < 1.5
    cds[8] = 0.5 < x < 1.5 && -1.5 < y < -0.5
    cds[9] = 1.5 < x < 2.5 && 1.5 < y < 2.5
    cds[10] = 1.5 < x < 2.5 && -0.5 < y < 0.5
    cds[11] = 1.5 < x < 2.5 && -2.5 < y < -1.5

    if any(cds) == true
        return true
    else
        return false
    end
end

begin
    σs = zeros(ps.nx, ps.ny)
    σa = zeros(ps.nx, ps.ny)
    for i = 1:ps.nx, j = 1:ps.ny
        if is_absorb(ps.x[i, j], ps.y[i, j])
            σs[i, j] = 0.0
            σa[i, j] = 10.0
        else
            σs[i, j] = 1.0
            σa[i, j] = 0.0
        end
    end
    σt = σs + σa
    σq = zeros(ps.nx, ps.ny)
    for i = 1:ps.nx, j = 1:ps.ny
        if -0.5 < ps.x[i, j] < 0.5 && -0.5 < ps.y[i, j] < 0.5
            σq[i, j] = 30.0 / (4.0 * π)
        else
            σq[i, j] = 0.0
        end
    end
end

u0 = zeros(ps.nx, ps.ny, vs.nu, nsp, nsp)
for i = 1:ps.nx, j = 1:ps.ny, p = 1:nsp, q = 1:nsp
    u0[i, j, :, p, q] .= 1e-6
end

function mol!(du, u, p, t)
    dx,
    dy,
    velo,
    weights,
    δu,
    δv,
    fx,
    fy,
    ux_face,
    uy_face,
    fx_face,
    fy_face,
    fx_interaction,
    fy_interaction,
    rhs1,
    rhs2,
    σs,
    σt,
    σq,
    ll,
    lr,
    lpdm,
    dgl,
    dgr = p

    nx = size(u, 1)
    ny = size(u, 2)
    nu = size(u, 3)
    nsp = size(u, 4)

    @inbounds @threads for q = 1:nsp
        for p = 1:nsp, k = 1:nu, j = 1:ny, i = 1:nx
            Jx = 0.5 * dx[i, j]
            fx[i, j, k, p, q] = velo[k, 1] * u[i, j, k, p, q] / Jx
        end
    end
    @inbounds @threads for q = 1:nsp
        for p = 1:nsp, k = 1:nu, j = 1:ny, i = 1:nx
            Jy = 0.5 * dy[i, j]
            fy[i, j, k, p, q] = velo[k, 2] * u[i, j, k, p, q] / Jy
        end
    end

    @inbounds @threads for q = 1:nsp
        for k = 1:nu, j = 1:ny, i = 1:nx
            ux_face[i, j, k, q, 1] = dot(u[i, j, k, :, q], ll)
            ux_face[i, j, k, q, 2] = dot(u[i, j, k, :, q], lr)

            fx_face[i, j, k, q, 1] = dot(fx[i, j, k, :, q], ll)
            fx_face[i, j, k, q, 2] = dot(fx[i, j, k, :, q], lr)
        end
    end
    @inbounds @threads for p = 1:nsp
        for k = 1:nu, j = 1:ny, i = 1:nx
            uy_face[i, j, k, p, 1] = dot(u[i, j, k, p, :], ll)
            uy_face[i, j, k, p, 2] = dot(u[i, j, k, p, :], lr)

            fy_face[i, j, k, p, 1] = dot(fy[i, j, k, p, :], ll)
            fy_face[i, j, k, p, 2] = dot(fy[i, j, k, p, :], lr)
        end
    end

    @inbounds @threads for k = 1:nsp
        for j = 1:ny, i = 2:nx
            @. fx_interaction[i, j, :, k] =
                fx_face[i-1, j, :, k, 2] * δu + fx_face[i, j, :, k, 1] * (1.0 - δu)
        end
    end
    @inbounds @threads for k = 1:nsp
        for i = 1:nx, j = 2:ny
            @. fy_interaction[i, j, :, k] =
                fy_face[i, j-1, :, k, 2] * δv + fy_face[i, j, :, k, 1] * (1.0 - δv)
        end
    end

    @inbounds @threads for q = 1:nsp
        for p = 1:nsp, k = 1:nu, j = 1:ny, i = 1:nx
            rhs1[i, j, k, p, q] = dot(fx[i, j, k, :, q], lpdm[p, :])
        end
    end
    @inbounds @threads for q = 1:nsp
        for p = 1:nsp, k = 1:nu, j = 1:ny, i = 1:nx
            rhs2[i, j, k, p, q] = dot(fy[i, j, k, p, :], lpdm[q, :])
        end
    end

    @inbounds @threads for q = 1:nsp
        for p = 1:nsp, j = 2:ny-1, i = 2:nx-1
            M = discrete_moments(u[i, j, :, p, q], weights) / 4 / π

            for k = 1:nu
                du[i, j, k, p, q] =
                    -(
                        rhs1[i, j, k, p, q] +
                        rhs2[i, j, k, p, q] +
                        (fx_interaction[i, j, k, q] - fx_face[i, j, k, q, 1]) * dgl[p] +
                        (fx_interaction[i+1, j, k, q] - fx_face[i, j, k, q, 2]) * dgr[p] +
                        (fy_interaction[i, j, k, p] - fy_face[i, j, k, p, 1]) * dgl[q] +
                        (fy_interaction[i, j+1, k, p] - fy_face[i, j, k, p, 2]) * dgr[q]
                    ) + σs[i, j] * M - σt[i, j] * u[i, j, k, p, q] + σq[i, j]
            end
        end
    end
end

tspan = (0.0, set.maxTime)
dt = set.cfl * minimum(ps.dx) / (vs.u1 + 2)
nt = floor(tspan[2] / dt) |> Int

begin
    fx = zeros(ps.nx, ps.ny, vs.nu, nsp, nsp)
    fy = zero(fx)
    ux_face = zeros(ps.nx, ps.ny, vs.nu, nsp, 2)
    uy_face = zero(ux_face)
    fx_face = zero(ux_face)
    fy_face = zero(ux_face)
    fx_interaction = zeros(ps.nx, ps.ny, vs.nu, nsp)
    fy_interaction = zero(fx_interaction)
    rhs1 = zero(u0)
    rhs2 = zero(u0)
end
p = (
    ps.dx,
    ps.dy,
    vs.u,
    vs.weights,
    δu,
    δv,
    fx,
    fy,
    ux_face,
    uy_face,
    fx_face,
    fy_face,
    fx_interaction,
    fy_interaction,
    rhs1,
    rhs2,
    σs,
    σt,
    σq,
    ps.ll,
    ps.lr,
    ps.dl,
    ps.dhl,
    ps.dhr,
)

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

@showprogress for iter = 1:nt*500
    step!(itg)
end

begin
    coord = zeros(ps.nx * nsp, ps.ny * nsp, 2)
    sol = zeros(ps.nx * nsp, ps.ny * nsp, 3)
    for i = 1:ps.nx, j = 1:ps.ny
        idx0 = (i - 1) * nsp
        idy0 = (j - 1) * nsp

        for k = 1:nsp, l = 1:nsp
            idx = idx0 + k
            idy = idy0 + l
            coord[idx, idy, 1] = ps.xpg[i, j, k, l, 1]
            coord[idx, idy, 2] = ps.xpg[i, j, k, l, 2]
            sol[idx, idy, 1] = sum(itg.u[i, j, :, k, l] .* vs.weights)
        end
    end

    x_uni =
        coord[1, 1, 1]:(coord[end, 1, 1]-coord[1, 1, 1])/(ps.nx*nsp-1):coord[end, 1, 1] |>
        collect
    y_uni =
        coord[1, 1, 2]:(coord[1, end, 2]-coord[1, 1, 2])/(ps.ny*nsp-1):coord[1, end, 2] |>
        collect
    n_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], sol[:, :, 1], kind = "cubic")
    n_uni = n_ref(x_uni, y_uni)
end
#pic = contourf(x_uni, y_uni, sol[:, :, 1]')

begin
    close("all")
    fig = PyPlot.figure("contour", figsize = (6.5, 5))
    PyPlot.contourf(
        x_uni,
        y_uni,
        n_uni',
        linewidth = 1,
        levels = 20,
        cmap = PyPlot.ColorMap("inferno"),
    )
    PyPlot.colorbar()
    PyPlot.xlabel("x")
    PyPlot.ylabel("y")
    #PyPlot.title("U-velocity")
    #PyPlot.xlim(0.01,0.99)
    #PyPlot.ylim(0.01,0.99)
    #PyPlot.grid("on")
    display(fig)
end

#fig.savefig("cavity_u.pdf")
