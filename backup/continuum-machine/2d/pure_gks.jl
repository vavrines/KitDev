using Kinetic, Plots, LinearAlgebra
using ProgressMeter: @showprogress

cd(@__DIR__)
D = Dict{Symbol,Any}()
begin
    D[:matter] = "gas"
    D[:case] = "cavity"
    D[:space] = "2d2f2v"
    D[:flux] = "kfvs"
    D[:collision] = "bgk"
    D[:nSpecies] = 1
    D[:interpOrder] = 2
    D[:limiter] = "vanleer"
    D[:boundary] = "maxwell"
    D[:cfl] = 0.3
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
    D[:nu] = 8
    D[:vmin] = -5.0
    D[:vmax] = 5.0
    D[:nv] = 8
    D[:vMeshType] = "rectangle"
    D[:nug] = 0
    D[:nvg] = 0

    D[:knudsen] = 0.005
    D[:mach] = 0.0
    D[:prandtl] = 1.0
    D[:inK] = 1.0
    D[:omega] = 0.81
    D[:alphaRef] = 1.0
    D[:omegaRef] = 0.5

    D[:uLid] = 0.15
    D[:vLid] = 0.0
    D[:tLid] = 1.0
end

ks = SolverSet(D)
ctr, a1face, a2face = init_fvm(ks, ks.ps, :dynamic_array; structarray = true)

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ks.ib.wL)

@showprogress for iter = 1:200#nt
    #reconstruct!(ks, ctr)
    #evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))

    # horizontal flux
    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        for i = 2:ks.pSpace.nx
            flux_gks!(
                a1face[i, j].fw,
                ctr[i-1, j].w .+ ctr[i-1, j].sw[:, 1] .* ks.ps.dx[i-1, j] / 2,
                ctr[i, j].w .- ctr[i, j].sw[:, 1] .* ks.ps.dx[i, j] / 2,
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
        end
    end

    # vertical flux
    @inbounds Threads.@threads for j = 2:ks.pSpace.ny
        for i = 1:ks.pSpace.nx
            wL = KitBase.local_frame(ctr[i, j-1].w, 0.0, 1.0)
            wR = KitBase.local_frame(ctr[i, j].w, 0.0, 1.0)
            swL = KitBase.local_frame(ctr[i, j-1].sw[:, 2], 0.0, 1.0)
            swR = KitBase.local_frame(ctr[i, j].sw[:, 2], 0.0, 1.0)

            flux_gks!(
                a2face[i, j].fw,
                wL .+ swL .* ks.ps.dy[i, j-1] / 2,
                wR .- swR .* ks.ps.dy[i, j] / 2,
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

            a2face[i, j].fw .= KitBase.global_frame(a2face[i, j].fw, 0.0, 1.0)
        end
    end

    # boundary flux
    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        KitBase.flux_boundary_maxwell!(
            a1face[1, j].fw,
            ks.ib.bcL,
            ctr[1, j].w,
            ks.gas.K,
            ks.gas.γ,
            dt,
            ctr[1, j].dy,
            1.0,
        )

        KitBase.flux_boundary_maxwell!(
            a1face[ks.pSpace.nx+1, j].fw,
            ks.ib.bcR,
            ctr[ks.pSpace.nx, j].w,
            ks.gas.K,
            ks.gas.γ,
            dt,
            ctr[ks.pSpace.nx, j].dy,
            -1.0,
        )
    end

    @inbounds Threads.@threads for i = 1:ks.pSpace.nx
        w = KitBase.local_frame(ctr[i, 1].w, 0.0, 1.0)
        KitBase.flux_boundary_maxwell!(
            a2face[i, 1].fw,
            ks.ib.bcD,
            w,
            ks.gas.K,
            ks.gas.γ,
            dt,
            ctr[i, 1].dx,
            1,
        )
        a2face[i, 1].fw .= KitBase.global_frame(a2face[i, 1].fw, 0.0, 1.0)

        w1 = KitBase.local_frame(ctr[i, ks.pSpace.ny].w, 0.0, 1.0)
        KitBase.flux_boundary_maxwell!(
            a2face[i, ks.pSpace.ny+1].fw,
            [1.0, 0.0, -0.05, 1.0],
            w1,
            ks.gas.K,
            ks.gas.γ,
            dt,
            ctr[i, ks.pSpace.ny].dy,
            -1,
        )
        a2face[i, ks.pSpace.ny+1].fw .=
            KitBase.global_frame(a2face[i, ks.pSpace.ny+1].fw, 0.0, 1.0)
    end

    #update!(ks, ctr, a1face, a2face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))

    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        for i = 1:ks.pSpace.nx
            @. ctr[i, j].w +=
                (
                    a1face[i, j].fw - a1face[i+1, j].fw + a2face[i, j].fw -
                    a2face[i, j+1].fw
                ) / ctr[i, j].dx / ctr[i, j].dy
            ctr[i, j].prim .= conserve_prim(ctr[i, j].w, ks.gas.γ)
        end
    end

    t += dt
end

plot_contour(ks, ctr)

###
# Sod along y direction
###

begin
    D[:matter] = "gas"
    D[:case] = "cavity"
    D[:space] = "2d2f2v"
    D[:flux] = "kfvs"
    D[:collision] = "bgk"
    D[:nSpecies] = 1
    D[:interpOrder] = 2
    D[:limiter] = "vanleer"
    D[:boundary] = "maxwell"
    D[:cfl] = 0.3
    D[:maxTime] = 5.0

    D[:x0] = 0.0
    D[:x1] = 1.0
    D[:nx] = 1
    D[:y0] = 0.0
    D[:y1] = 1.0
    D[:ny] = 100
    D[:pMeshType] = "uniform"
    D[:nxg] = 0
    D[:nyg] = 0

    D[:umin] = -5.0
    D[:umax] = 5.0
    D[:nu] = 8
    D[:vmin] = -5.0
    D[:vmax] = 5.0
    D[:nv] = 8
    D[:vMeshType] = "rectangle"
    D[:nug] = 0
    D[:nvg] = 0

    D[:knudsen] = 0.001
    D[:mach] = 0.0
    D[:prandtl] = 1.0
    D[:inK] = 1.0
    D[:omega] = 0.81
    D[:alphaRef] = 1.0
    D[:omegaRef] = 0.5

    D[:uLid] = 0.15
    D[:vLid] = 0.0
    D[:tLid] = 1.0
end

ks = SolverSet(D)
KS = ks
ctr = Array{ControlVolume2D}(undef, 1, ks.ps.ny)
a1face = Array{Interface2D}(undef, KS.pSpace.nx + 1, KS.pSpace.ny)
a2face = Array{Interface2D}(undef, KS.pSpace.nx, KS.pSpace.ny + 1)

for j in axes(ctr, 2), i in axes(ctr, 1)
    if j <= KS.pSpace.ny ÷ 2
        prim = [1.0, 0.0, 0.0, 0.5]
        w = conserve_prim(prim, ks.gas.γ)

        ctr[i, j] = ControlVolume2D(
            KS.pSpace.x[i, j],
            KS.pSpace.y[i, j],
            KS.pSpace.dx[i, j],
            KS.pSpace.dy[i, j],
            w,
            prim,
        )
    else
        prim = [0.125, 0.0, 0.0, 0.625]
        w = conserve_prim(prim, ks.gas.γ)

        ctr[i, j] = ControlVolume2D(
            KS.pSpace.x[i, j],
            KS.pSpace.y[i, j],
            KS.pSpace.dx[i, j],
            KS.pSpace.dy[i, j],
            w,
            prim,
        )
    end
end

for i = 1:KS.pSpace.nx
    for j = 1:KS.pSpace.ny
        a2face[i, j] = Interface2D(KS.pSpace.dx[i, j], 0.0, 1.0, KS.ib.wL)
    end
    a2face[i, KS.pSpace.ny+1] =
        Interface2D(KS.pSpace.dx[i, KS.pSpace.ny], 0.0, 1.0, KS.ib.wL)
end

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(0.2 ÷ dt) + 1
res = zero(ks.ib.wL)

@showprogress for iter = 1:nt
    #reconstruct!(ks, ctr)
    #evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))

    # horizontal flux
    #=@inbounds Threads.@threads for j = 1:ks.pSpace.ny
        for i = 2:ks.pSpace.nx
            flux_gks!(
                a1face[i, j].fw,
                ctr[i-1, j].w .+ ctr[i-1, j].sw[:, 1] .* ks.ps.dx[i-1, j]/2,
                ctr[i, j].w .- ctr[i, j].sw[:, 1] .* ks.ps.dx[i, j]/2,
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                dt,
                ks.ps.dx[i-1, j]/2,
                ks.ps.dx[i, j]/2,
                a1face[i, j].len,
                ctr[i-1, j].sw[:, 1],
                ctr[i, j].sw[:, 1],
            )
        end
    end=#

    # vertical flux
    @inbounds Threads.@threads for j = 2:ks.pSpace.ny
        for i = 1:ks.pSpace.nx
            wL = local_frame(ctr[i, j-1].w, 0.0, 1.0)
            wR = local_frame(ctr[i, j].w, 0.0, 1.0)
            swL = local_frame(ctr[i, j-1].sw[:, 2], 0.0, 1.0)
            swR = local_frame(ctr[i, j].sw[:, 2], 0.0, 1.0)
            #=
                        flux_gks!(
                            a2face[i, j].fw,
                            wL .+ swL .* ks.ps.dy[i, j-1]/2,
                            wR .- swR .* ks.ps.dy[i, j]/2,
                            ks.gas.K,
                            ks.gas.γ,
                            ks.gas.μᵣ,
                            ks.gas.ω,
                            dt,
                            ks.ps.dy[i, j-1]/2,
                            ks.ps.dy[i, j]/2,
                            a2face[i, j].len,
                            swL,
                            swR,
                        )=#

            flux_roe!(a2face[i, j].fw, wL, wR, ks.gas.γ, dt)

            a2face[i, j].fw .=
                KitBase.global_frame(a2face[i, j].fw, 0.0, 1.0) .* ctr[i, j].dx
        end
    end

    #update!(ks, ctr, a1face, a2face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))

    @inbounds Threads.@threads for j = 2:ks.pSpace.ny-1
        for i = 1:ks.pSpace.nx
            #@. ctr[i, j].w += (a1face[i, j].fw - a1face[i+1, j].fw + a2face[i, j].fw - a2face[i, j+1].fw) / ctr[i, j].dx / ctr[i, j].dy
            @. ctr[i, j].w +=
                (a2face[i, j].fw - a2face[i, j+1].fw) / ctr[i, j].dx / ctr[i, j].dy
            ctr[i, j].prim .= conserve_prim(ctr[i, j].w, ks.gas.γ)
        end
    end

    t += dt
end

begin
    sol = zeros(ks.ps.ny, 4)
    for i in axes(sol, 1)
        sol[i, :] .= ctr[1, i].prim
        sol[i, 4] = 1 / sol[i, 4]
    end

    plot(ks.ps.y[:], sol[:, :])
end



###
# Sod along x direction
###

begin
    D[:matter] = "gas"
    D[:case] = "cavity"
    D[:space] = "2d2f2v"
    D[:flux] = "kfvs"
    D[:collision] = "bgk"
    D[:nSpecies] = 1
    D[:interpOrder] = 2
    D[:limiter] = "vanleer"
    D[:boundary] = "maxwell"
    D[:cfl] = 0.3
    D[:maxTime] = 5.0

    D[:x0] = 0.0
    D[:x1] = 1.0
    D[:nx] = 100
    D[:y0] = 0.0
    D[:y1] = 1.0
    D[:ny] = 1
    D[:pMeshType] = "uniform"
    D[:nxg] = 0
    D[:nyg] = 0

    D[:umin] = -5.0
    D[:umax] = 5.0
    D[:nu] = 8
    D[:vmin] = -5.0
    D[:vmax] = 5.0
    D[:nv] = 8
    D[:vMeshType] = "rectangle"
    D[:nug] = 0
    D[:nvg] = 0

    D[:knudsen] = 0.001
    D[:mach] = 0.0
    D[:prandtl] = 1.0
    D[:inK] = 1.0
    D[:omega] = 0.81
    D[:alphaRef] = 1.0
    D[:omegaRef] = 0.5

    D[:uLid] = 0.15
    D[:vLid] = 0.0
    D[:tLid] = 1.0
end

ks = SolverSet(D)
KS = ks
ctr = Array{ControlVolume2D}(undef, ks.ps.nx, 1)
a1face = Array{Interface2D}(undef, KS.pSpace.nx + 1, KS.pSpace.ny)

for j in axes(ctr, 2), i in axes(ctr, 1)
    if i <= KS.pSpace.nx ÷ 2
        prim = [1.0, 0.0, 0.0, 0.5]
        w = conserve_prim(prim, ks.gas.γ)

        ctr[i, j] = ControlVolume2D(
            KS.pSpace.x[i, j],
            KS.pSpace.y[i, j],
            KS.pSpace.dx[i, j],
            KS.pSpace.dy[i, j],
            w,
            prim,
        )
    else
        prim = [0.125, 0.0, 0.0, 0.625]
        w = conserve_prim(prim, ks.gas.γ)

        ctr[i, j] = ControlVolume2D(
            KS.pSpace.x[i, j],
            KS.pSpace.y[i, j],
            KS.pSpace.dx[i, j],
            KS.pSpace.dy[i, j],
            w,
            prim,
        )
    end
end

for i = 1:KS.pSpace.nx
    for j = 1:KS.pSpace.ny
        a1face[i, j] = Interface2D(KS.pSpace.dx[i, j], 1.0, 0.0, KS.ib.wL)
    end
    a1face[ks.ps.nx+1, 1] = Interface2D(KS.pSpace.dx[i, 1], 1.0, 0.0, KS.ib.wL)
end

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(0.2 ÷ dt) + 1
res = zero(ks.ib.wL)

@showprogress for iter = 1:nt
    #reconstruct!(ks, ctr)
    #evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))

    # horizontal flux
    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        for i = 2:ks.pSpace.nx
            #=flux_gks!(
                a1face[i, j].fw,
                ctr[i-1, j].w .+ ctr[i-1, j].sw[:, 1] .* ks.ps.dx[i-1, j]/2,
                ctr[i, j].w .- ctr[i, j].sw[:, 1] .* ks.ps.dx[i, j]/2,
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                dt,
                ks.ps.dx[i-1, j]/2,
                ks.ps.dx[i, j]/2,
                a1face[i, j].len,
                ctr[i-1, j].sw[:, 1],
                ctr[i, j].sw[:, 1],
            )=#

            flux_roe!(a1face[i, j].fw, ctr[i-1, j].w, ctr[i, j].w, ks.gas.γ, dt)
            a1face[i, j].fw .*= ctr[i, j].dy
        end
    end

    #update!(ks, ctr, a1face, a2face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))

    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        for i = 2:ks.pSpace.nx-1
            #@. ctr[i, j].w += (a1face[i, j].fw - a1face[i+1, j].fw + a2face[i, j].fw - a2face[i, j+1].fw) / ctr[i, j].dx / ctr[i, j].dy
            @. ctr[i, j].w +=
                (a1face[i, j].fw - a1face[i+1, j].fw) / ctr[i, j].dx / ctr[i, j].dy
            ctr[i, j].prim .= conserve_prim(ctr[i, j].w, ks.gas.γ)
        end
    end

    t += dt
end

begin
    sol = zeros(ks.ps.nx, 4)
    for i in axes(sol, 1)
        sol[i, :] .= ctr[i, 1].prim
        sol[i, 4] = 1 / sol[i, 4]
    end

    plot(ks.ps.x[:], sol[:, :])
end
