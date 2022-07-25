using Kinetic, Plots, LinearAlgebra, JLD2, Flux
using ProgressMeter: @showprogress
using Flux: onecold

cd(@__DIR__)
D = Dict{Symbol,Any}()
begin
    D[:matter] = "gas"
    D[:case] = "cavity"
    D[:space] = "2d1f2v"
    D[:flux] = "kfvs"
    D[:collision] = "bgk"
    D[:nSpecies] = 1
    D[:interpOrder] = 2
    D[:limiter] = "vanleer"
    D[:boundary] = "maxwell"
    D[:cfl] = 0.5
    D[:maxTime] = 5.0

    D[:x0] = 0.0
    D[:x1] = 1.0
    D[:nx] = 41
    D[:y0] = 0.0
    D[:y1] = 1.0
    D[:ny] = 41
    D[:pMeshType] = "uniform"
    D[:nxg] = 1
    D[:nyg] = 1

    D[:umin] = -5.0
    D[:umax] = 5.0
    D[:nu] = 24
    D[:vmin] = -5.0
    D[:vmax] = 5.0
    D[:nv] = 24
    D[:vMeshType] = "rectangle"
    D[:nug] = 0
    D[:nvg] = 0

    D[:knudsen] = 0.005
    D[:mach] = 0.0
    D[:prandtl] = 1.0
    D[:inK] = 0.0
    D[:omega] = 0.81
    D[:alphaRef] = 1.0
    D[:omegaRef] = 0.5

    D[:uLid] = 0.15
    D[:vLid] = 0.0
    D[:tLid] = 1.0
end

ks = SolverSet(D)
ctr, a1face, a2face = init_fvm(ks, ks.ps, :dynamic_array; structarray = true)

@load "nn.jld2" nn

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ks.ib.wL)

@showprogress for iter = 1:50#nt
    reconstruct!(ks, ctr)
    #evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))

    # horizontal flux
    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        for i = 2:ks.pSpace.nx
            w = (ctr[i-1, j].w .+ ctr[i, j].w) ./ 2
            sw = (ctr[i-1, j].sw .+ ctr[i, j].sw) ./ 2
            gra = (sw[:, 1] .^ 2 + sw[:, 2] .^ 2) .^ 0.5
            prim = conserve_prim(w, ks.gas.γ)
            tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
            regime = nn([w; gra; tau]) |> onecold

            if regime == 1
                flux_gks!(
                    a1face[i, j].fw,
                    a1face[i, j].ff,
                    ctr[i-1, j].w .+ ctr[i-1, j].sw[:, 1] .* ks.ps.dx[i-1, j] / 2,
                    ctr[i, j].w .- ctr[i, j].sw[:, 1] .* ks.ps.dx[i, j] / 2,
                    ks.vSpace.u,
                    ks.vSpace.v,
                    ks.gas.K,
                    ks.gas.γ,
                    ks.gas.μᵣ,
                    ks.gas.ω,
                    dt,
                    ks.ps.dx[i-1, j] / 2,
                    ks.ps.dx[i, j] / 2,
                    a1face[i, j].len,
                    ctr[i-1, j].sw[:, 1],
                    ctr[i, j].sw[:, 1],
                )
            elseif regime == 2
                flux_kfvs!(
                    a1face[i, j].fw,
                    a1face[i, j].ff,
                    ctr[i-1, j].f,
                    ctr[i, j].f,
                    ks.vSpace.u,
                    ks.vSpace.v,
                    ks.vSpace.weights,
                    dt,
                    a1face[i, j].len,
                )
            end
        end
    end

    # vertical flux
    vn = ks.vSpace.v
    vt = -ks.vSpace.u
    @inbounds Threads.@threads for j = 2:ks.pSpace.ny
        for i = 1:ks.pSpace.nx
            w = (ctr[i, j-1].w .+ ctr[i, j].w) ./ 2
            sw = (ctr[i, j-1].sw .+ ctr[i, j].sw) ./ 2
            gra = (sw[:, 1] .^ 2 + sw[:, 2] .^ 2) .^ 0.5
            prim = conserve_prim(w, ks.gas.γ)
            tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
            regime = nn([w; gra; tau]) |> onecold

            wL = KitBase.local_frame(ctr[i, j-1].w, 0.0, 1.0)
            wR = KitBase.local_frame(ctr[i, j].w, 0.0, 1.0)
            swL = KitBase.local_frame(ctr[i, j-1].sw[:, 2], 0.0, 1.0)
            swR = KitBase.local_frame(ctr[i, j].sw[:, 2], 0.0, 1.0)

            if regime == 1
                flux_gks!(
                    a2face[i, j].fw,
                    a2face[i, j].ff,
                    wL .+ swL .* ks.ps.dy[i, j-1] / 2,
                    wR .- swR .* ks.ps.dy[i, j] / 2,
                    vn,
                    vt,
                    ks.gas.K,
                    ks.gas.γ,
                    ks.gas.μᵣ,
                    ks.gas.ω,
                    dt,
                    ks.ps.dy[i, j-1] / 2,
                    ks.ps.dy[i, j] / 2,
                    a2face[i, j].len,
                    swL,
                    swR,
                )
            elseif regime == 2
                KitBase.flux_kfvs!(
                    a2face[i, j].fw,
                    a2face[i, j].ff,
                    ctr[i, j-1].f,
                    ctr[i, j].f,
                    vn,
                    vt,
                    ks.vSpace.weights,
                    dt,
                    a2face[i, j].len,
                )
            end

            a2face[i, j].fw .= KitBase.global_frame(a2face[i, j].fw, 0.0, 1.0)
        end
    end

    # boundary flux
    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        KitBase.flux_boundary_maxwell!(
            a1face[1, j].fw,
            a1face[1, j].ff,
            ks.ib.bcL,
            ctr[1, j].f,
            ks.vSpace.u,
            ks.vSpace.v,
            ks.vSpace.weights,
            dt,
            ctr[1, j].dy,
            1.0,
        )

        KitBase.flux_boundary_maxwell!(
            a1face[ks.pSpace.nx+1, j].fw,
            a1face[ks.pSpace.nx+1, j].ff,
            ks.ib.bcR,
            ctr[ks.pSpace.nx, j].f,
            ks.vSpace.u,
            ks.vSpace.v,
            ks.vSpace.weights,
            dt,
            ctr[ks.pSpace.nx, j].dy,
            -1.0,
        )
    end

    @inbounds Threads.@threads for i = 1:ks.pSpace.nx
        KitBase.flux_boundary_maxwell!(
            a2face[i, 1].fw,
            a2face[i, 1].ff,
            ks.ib.bcD,
            ctr[i, 1].f,
            vn,
            vt,
            ks.vSpace.weights,
            dt,
            ctr[i, 1].dx,
            1,
        )
        a2face[i, 1].fw .= KitBase.global_frame(a2face[i, 1].fw, 0.0, 1.0)

        KitBase.flux_boundary_maxwell!(
            a2face[i, ks.pSpace.ny+1].fw,
            a2face[i, ks.pSpace.ny+1].ff,
            [1.0, 0.0, -0.15, 1.0],
            ctr[i, ks.pSpace.ny].f,
            vn,
            vt,
            ks.vSpace.weights,
            dt,
            ctr[i, ks.pSpace.ny].dy,
            -1,
        )
        a2face[i, ks.pSpace.ny+1].fw .=
            KitBase.global_frame(a2face[i, ks.pSpace.ny+1].fw, 0.0, 1.0)
    end

    update!(
        ks,
        ctr,
        a1face,
        a2face,
        dt,
        res;
        coll = Symbol(ks.set.collision),
        bc = Symbol(ks.set.boundary),
    )

    t += dt
end

plot_contour(ks, ctr)

# detector
for j = 1:ks.pSpace.ny
    for i = 2:ks.pSpace.nx
        w = (ctr[i-1, j].w .+ ctr[i, j].w) ./ 2
        sw = (ctr[i-1, j].sw .+ ctr[i, j].sw) ./ 2
        gra = (sw[:, 1] .^ 2 + sw[:, 2] .^ 2) .^ 0.5
        prim = conserve_prim(w, ks.gas.γ)
        tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
        regime = nn([w; gra; tau]) |> onecold

        if regime == 1
            @show "here"
        end
    end
end
