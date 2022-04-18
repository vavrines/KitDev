using Kinetic, Plots, LinearAlgebra, JLD2, Flux
using KitBase.ProgressMeter: @showprogress
using Flux: onecold
pyplot()

cd(@__DIR__)
include("tools.jl")

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

function rg!(KS, ctr, cig)
    nx, ny, dx, dy = begin
        if KS.ps isa CSpace2D
            KS.ps.nr, KS.ps.nθ, KS.ps.dr, KS.ps.darc
        else
            KS.ps.nx, KS.ps.ny, KS.ps.dx, KS.ps.dy
        end
    end

    @inbounds Threads.@threads for j = 1:ny
        for i = 2:nx
            swx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (KS.ps.x[i+1, j] - KS.ps.x[i-1, j])
            swy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (KS.ps.y[i+1, j] - KS.ps.y[i-1, j])
            swx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (KS.ps.x[i, j+1] - KS.ps.x[i, j-1])
            swy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (KS.ps.y[i, j+1] - KS.ps.y[i, j-1])
            swx = (swx1 + swx2) ./ 2
            swy = (swy1 + swy2) ./ 2
            #sw = sqrt.(swx.^2 + swy.^2)

            


            w = (ctr[i-1, j].w + ctr[i, j].w) ./ 2
            f = (ctr[i-1, j].h + ctr[i, j].h) ./ 2
            swx = (ctr[i, j].w - ctr[i-1, j].w) / (ks.ps.x[i, j] - ks.ps.x[i-1, j])
            swy = (ctr[i, j].w - ctr[i-1, j].w) / (ks.ps.y[i, j] - ks.ps.y[i-1, j])
            #sw = sqrt.(sw1.^2 + sw2.^2)
            prim = conserve_prim(w, ks.gas.γ)
            #τ = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
            regime = judge_regime(ks, f, prim, swx, swy)

            if regime == 2
                flux_kfvs!(
                    a1face[i, j].fw,
                    a1face[i, j].fh,
                    a1face[i, j].fb,
                    ctr[i-1, j].h .+ 0.5 .* dx[i-1, j] .* ctr[i-1, j].sh[:, :, 1],
                    ctr[i-1, j].b .+ 0.5 .* dx[i-1, j] .* ctr[i-1, j].sb[:, :, 1],
                    ctr[i, j].h .- 0.5 .* dx[i, j] .* ctr[i, j].sh[:, :, 1],
                    ctr[i, j].b .- 0.5 .* dx[i, j] .* ctr[i, j].sb[:, :, 1],
                    vn,
                    vt,
                    KS.vSpace.weights,
                    dt,
                    a1face[i, j].len,
                    ctr[i-1, j].sh[:, :, 1],
                    ctr[i-1, j].sb[:, :, 1],
                    ctr[i, j].sh[:, :, 1],
                    ctr[i, j].sb[:, :, 1],
                )
            else
                wL = local_frame(ctr[i-1, j].w, a1face[i, j].n[1], a1face[i, j].n[2])
                wR = local_frame(ctr[i, j].w, a1face[i, j].n[1], a1face[i, j].n[2])
                swL = local_frame(ctr[i-1, j].sw[:, 1], a1face[i, j].n[1], a1face[i, j].n[2])
                swR = local_frame(ctr[i, j].sw[:, 1], a1face[i, j].n[1], a1face[i, j].n[2])

                flux_gks!(
                    a1face[i, j].fw,
                    a1face[i, j].fh,
                    a1face[i, j].fb,
                    wL .+ 0.5 .* dx[i, j] .* swL,
                    wR .- 0.5 .* dx[i, j] .* swR,
                    vn,
                    vt,
                    ks.gas.K,
                    ks.gas.γ,
                    ks.gas.μᵣ,
                    ks.gas.ω,
                    dt,
                    0.5 .* dx[i-1, j],
                    0.5 .* dx[i, j],
                    a1face[i, j].len,
                    swL,
                    swR,
                )
            end

            a1face[i, j].fw .=
                global_frame(a1face[i, j].fw, a1face[i, j].n[1], a1face[i, j].n[2])
        end
    end


end