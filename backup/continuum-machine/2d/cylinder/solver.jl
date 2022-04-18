using Kinetic, Plots, LinearAlgebra, JLD2, Flux
using KitBase.ProgressMeter: @showprogress
using Flux: onecold
pyplot()

cd(@__DIR__)
include("recon.jl")

function recon_pdf(ks, prim, swx, swy)
    Mu, Mv, Mxi, _, _1 = gauss_moments(prim, ks.gas.K)
    a = pdf_slope(prim, swx, ks.gas.K)
    b = pdf_slope(prim, swy, ks.gas.K)
    sw = -prim[1] .* (moments_conserve_slope(a, Mu, Mv, Mxi, 1, 0) .+ moments_conserve_slope(b, Mu, Mv, Mxi, 0, 1))
    A = pdf_slope(prim, sw, ks.gas.K)
    tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)

    return chapman_enskog(ks.vs.u, ks.vs.v, prim, a, b, A, tau)
end

begin
    set = Setup(
        case = "cylinder",
        space = "2d2f2v",
        boundary = ["maxwell", "extra", "mirror", "mirror"],
        limiter = "minmod",
        cfl = 0.5,
        maxTime = 10.0, # time
    )
    ps = CSpace2D(1.0, 6.0, 30, 0.0, π, 50, 1, 1)
    vs = VSpace2D(-10.0, 10.0, 48, -10.0, 10.0, 48)
    gas = Gas(Kn = 1e-2, Ma = 5.0, K = 1.0)
    
    prim0 = [1.0, 0.0, 0.0, 1.0]
    prim1 = [1.0, gas.Ma * sound_speed(1.0, gas.γ), 0.0, 1.0]
    fw = (args...) -> prim_conserve(prim1, gas.γ)
    ff = function(args...)
        prim = conserve_prim(fw(args...), gas.γ)
        h = maxwellian(vs.u, vs.v, prim)
        b = h .* gas.K / 2 / prim[end]
        return h, b
    end
    bc = function(x, y)
        if abs(x^2 + y^2 - 1) < 1e-3
            return prim0
        else
            return prim1
        end
    end
    ib = IB2F(fw, ff, bc)

    ks = SolverSet(set, ps, vs, gas, ib)
end

ctr, a1face, a2face = init_fvm(ks, ks.ps)
@load "kn3.jld2" ctr
@load "../nn.jld2" nn

begin
    sol = zeros(ks.ps.nr, ks.ps.nθ, 4)
    for i in axes(sol, 1), j in axes(sol, 2)
        sol[i, j, :] .= ctr[i, j].prim
        sol[i, j, end] = 1 / sol[i, j, end]
    end
    contourf(
        ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
        ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
        sol[:, :, 4],
        ratio = 1,
    )
end

# detector
rmap = zeros(ks.pSpace.nr, ks.pSpace.nθ)
for j = 1:ks.pSpace.nθ
    for i = 2:ks.pSpace.nr
        dx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
        dy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
        dx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
        dy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (ks.ps.y[i, j+1] - ks.ps.y[i, j-1])

        cs = (dx1, dx2) -> (dx1 + dx2) ./ 2
        swx = cs(dx1, dx2)
        swy = cs(dy1, dy2)
        sw = [sqrt(swx[i]^2 + swy[i]^2) for i in eachindex(swx)]

        τ = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω) * 10

        din = [ctr[i, j].w; sw; τ]

        w = (ctr[i-1, j].w .+ ctr[i, j].w) ./ 2
        regime = nn(din) |> onecold

        rmap[i, j] = regime
    end
end

rmap = zeros(ks.pSpace.nr, ks.pSpace.nθ)
for j = 1:ks.pSpace.nθ
    rmap[1, j] = 2

    for i = 2:ks.pSpace.nr
        dx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
        dy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
        dx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
        dy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (ks.ps.y[i, j+1] - ks.ps.y[i, j-1])
        
        cs = (dx1, dx2) -> (dx1 + dx2) ./ 2
        swx = cs(dx1, dx2)
        swy = cs(dy1, dy2)

        prim = ctr[i, j].prim
        fr = recon_pdf(ks, prim, swx, swy)
        L = norm((ctr[i, j].h .- fr) ./ prim[1])

        rmap[i, j] = ifelse(L <= 0.005, 1, 2)
    end
end

contourf(
    ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
    ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
    rmap,
    ratio = 1,
)




