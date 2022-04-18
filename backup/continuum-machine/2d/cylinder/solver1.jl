using Kinetic, Plots, LinearAlgebra, JLD2, Flux
using KitBase.ProgressMeter: @showprogress
using Flux: onecold
pyplot()

cd(@__DIR__)
include("tools.jl")

function ev!(
    KS::SolverSet,
    ctr::T1,
    a1face::T2,
    a2face::T2,
    dt;
    mode = KitBase.symbolize(KS.set.flux)::Symbol,
    bc = KitBase.symbolize(KS.set.boundary),
) where {T1<:AbstractArray{ControlVolume2D2F,2},T2<:AbstractArray{Interface2D2F,2}}

    nx, ny, dx, dy = begin
        if KS.ps isa CSpace2D
            KS.ps.nr, KS.ps.nθ, KS.ps.dr, KS.ps.darc
        else
            KS.ps.nx, KS.ps.ny, KS.ps.dx, KS.ps.dy
        end
    end

    if firstindex(KS.pSpace.x[:, 1]) < 1
        idx0 = 1
        idx1 = nx + 1
    else
        idx0 = 2
        idx1 = nx
    end
    if firstindex(KS.pSpace.y[1, :]) < 1
        idy0 = 1
        idy1 = ny + 1
    else
        idy0 = 2
        idy1 = ny
    end

    # x direction
    @inbounds Threads.@threads for j = 1:ny
        for i = idx0:idx1
            vn = KS.vSpace.u .* a1face[i, j].n[1] .+ KS.vSpace.v .* a1face[i, j].n[2]
            vt = KS.vSpace.v .* a1face[i, j].n[1] .- KS.vSpace.u .* a1face[i, j].n[2]

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

    # y direction
    @inbounds Threads.@threads for j = idy0:idy1
        for i = 1:nx
            vn = KS.vSpace.u .* a2face[i, j].n[1] .+ KS.vSpace.v .* a2face[i, j].n[2]
            vt = KS.vSpace.v .* a2face[i, j].n[1] .- KS.vSpace.u .* a2face[i, j].n[2]

            w = (ctr[i, j-1].w + ctr[i, j].w) ./ 2
            f = (ctr[i, j-1].h + ctr[i, j].h) ./ 2
            swx = (ctr[i, j].w - ctr[i, j-1].w) / (ks.ps.x[i, j] - ks.ps.x[i, j-1])
            swy = (ctr[i, j].w - ctr[i, j-1].w) / (ks.ps.y[i, j] - ks.ps.y[i, j-1])
            #sw = sqrt.(sw1.^2 + sw2.^2)
            prim = conserve_prim(w, ks.gas.γ)
            #τ = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
            regime = judge_regime(ks, f, prim, swx, swy)

            if regime == 2
                flux_kfvs!(
                    a2face[i, j].fw,
                    a2face[i, j].fh,
                    a2face[i, j].fb,
                    ctr[i, j-1].h .+ 0.5 .* dy[i, j-1] .* ctr[i, j-1].sh[:, :, 2],
                    ctr[i, j-1].b .+ 0.5 .* dy[i, j-1] .* ctr[i, j-1].sb[:, :, 2],
                    ctr[i, j].h .- 0.5 .* dy[i, j] .* ctr[i, j].sh[:, :, 2],
                    ctr[i, j].b .- 0.5 .* dy[i, j] .* ctr[i, j].sb[:, :, 2],
                    vn,
                    vt,
                    KS.vSpace.weights,
                    dt,
                    a2face[i, j].len,
                    ctr[i, j-1].sh[:, :, 2],
                    ctr[i, j-1].sb[:, :, 2],
                    ctr[i, j].sh[:, :, 2],
                    ctr[i, j].sb[:, :, 2],
                )
            else
                wL = local_frame(ctr[i, j-1].w, a2face[i, j].n[1], a2face[i, j].n[2])
                wR = local_frame(ctr[i, j].w, a2face[i, j].n[1], a2face[i, j].n[2])
                swL = local_frame(ctr[i, j-1].sw[:, 2], a2face[i, j].n[1], a2face[i, j].n[2])
                swR = local_frame(ctr[i, j].sw[:, 2], a2face[i, j].n[1], a2face[i, j].n[2])

                flux_gks!(
                    a2face[i, j].fw,
                    a2face[i, j].fh,
                    a2face[i, j].fb,
                    wL .+ 0.5 .* dy[i, j-1] .* swL,
                    wR .- 0.5 .* dy[i, j] .* swR,
                    vn,
                    vt,
                    ks.gas.K,
                    ks.gas.γ,
                    ks.gas.μᵣ,
                    ks.gas.ω,
                    dt,
                    0.5 .* dy[i, j-1],
                    0.5 .* dy[i, j],
                    a2face[i, j].len,
                    swL,
                    swR,
                )
            end

            a2face[i, j].fw .=
                global_frame(a2face[i, j].fw, a2face[i, j].n[1], a2face[i, j].n[2])
        end
    end

    KitBase.evolve_boundary!(KS, ctr, a1face, a2face, dt, mode, bc)

    return nothing

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

t = 0.0
dt = timestep(ks, ctr, 0.0)
nt = ks.set.maxTime ÷ dt |> Int
res = zeros(4)
@showprogress for iter = 1:100#nt
    #rc!(ks, ctr)
    ev!(ks, ctr, a1face, a2face, dt)
    #evolve!(ks, ctr, a1face, a2face, dt)
    update!(ks, ctr, a1face, a2face, dt, res)

    for j = ks.ps.nθ÷2+1:ks.ps.nθ
        ctr[ks.ps.nr+1, j].w .= ks.ib.fw(6, 0)
        ctr[ks.ps.nr+1, j].prim .= conserve_prim(ctr[ks.ps.nr+1, j].w, ks.gas.γ)
        ctr[ks.ps.nr+1, j].sw .= 0.0
        ctr[ks.ps.nr+1, j].h .= maxwellian(ks.vs.u, ks.vs.v, ctr[ks.ps.nr+1, j].prim)
        ctr[ks.ps.nr+1, j].b .= ctr[ks.ps.nr+1, j].h .* ks.gas.K ./ 2 ./ ctr[ks.ps.nr+1, j].prim[end]
    end

    global t += dt

    if maximum(res) < 1e-6
        break
    end
end

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
