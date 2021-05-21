using KitBase
using KitBase.Plots, KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)
D = read_dict("config.txt")

set = set_setup(D)
ps = set_geometry(D)
vs = set_velocity(D)
gas = set_property(D)

begin
    primL = [1.5, 0.0, 0.0, 1.0]
    wL = KitBase.prim_conserve(primL, gas.γ)
    hL = KitBase.maxwellian(vs.u, vs.v, primL)
    bL = @. hL * gas.K / 2 / primL[end]
    primR = [0.5, 0.0, 0.0, 1.0]
    wR = KitBase.prim_conserve(primR, gas.γ)
    hR = KitBase.maxwellian(vs.u, vs.v, primR)
    bR = @. hR * gas.K / 2 / primR[end]
    ib = KitBase.IB2F(
        wL,
        primL,
        hL,
        bL,
        primL,
        wL,
        primL,
        hL,
        bL,
        primR,
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    )
end

ks = KitBase.SolverSet(set, ps, vs, gas, ib, pwd())

ctr, a1face, a2face = KitBase.init_fvm(ks, ks.pSpace)
for i = 1:ks.pSpace.nx, j = 1:ks.pSpace.ny
    ctr[i, j].prim .= [1.0, 0.0, 0.0, 1.0]
    ctr[i, j].w .= KitBase.prim_conserve(ctr[i, j].prim, ks.gas.γ)
    ctr[i, j].h .= KitBase.maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
    ctr[i, j].b = @. ctr[i, j].h * ks.gas.K / 2 / ctr[i, j].prim[end]
end

dt = KitBase.timestep(ks, ctr, 0.0)
nt = ks.set.maxTime ÷ dt |> Int
res = zeros(4)

@showprogress for iter = 1:5000#nt
    # horizontal flux
    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        for i = 1:ks.pSpace.nx+1
            KitBase.flux_kfvs!(
                a1face[i, j].fw,
                a1face[i, j].fh,
                a1face[i, j].fb,
                ctr[i-1, j].h,
                ctr[i-1, j].b,
                ctr[i, j].h,
                ctr[i, j].b,
                ks.vSpace.u,
                ks.vSpace.v,
                ks.vSpace.weights,
                dt,
                a1face[i, j].len,
            )
        end
    end
    
    # vertical flux
    vn = ks.vSpace.v
    vt = -ks.vSpace.u
    @inbounds Threads.@threads for j = 2:ks.pSpace.ny
        for i = 1:ks.pSpace.nx
            KitBase.flux_kfvs!(
                a2face[i, j].fw,
                a2face[i, j].fh,
                a2face[i, j].fb,
                ctr[i, j-1].h,
                ctr[i, j-1].b,
                ctr[i, j].h,
                ctr[i, j].b,
                vn,
                vt,
                ks.vSpace.weights,
                dt,
                a2face[i, j].len,
            )
            a2face[i, j].fw .= KitBase.global_frame(a2face[i, j].fw, 0., 1.)
        end
    end
    
    # boundary flux    
    @inbounds Threads.@threads for i = 1:ks.pSpace.nx
        KitBase.flux_boundary_maxwell!(
            a2face[i, 1].fw,
            a2face[i, 1].fh,
            a2face[i, 1].fb,
            ks.ib.bcD,
            ctr[i, 1].h,
            ctr[i, 1].b,
            vn,
            vt,
            ks.vSpace.weights,
            ks.gas.K,
            dt,
            ctr[i, 1].dx,
            1,
        )
        a2face[i, 1].fw .= KitBase.global_frame(a2face[i, 1].fw, 0., 1.)
        
        KitBase.flux_boundary_maxwell!(
            a2face[i, ks.pSpace.ny+1].fw,
            a2face[i, ks.pSpace.ny+1].fh,
            a2face[i, ks.pSpace.ny+1].fb,
            ks.ib.bcU,
            ctr[i, ks.pSpace.ny].h,
            ctr[i, ks.pSpace.ny].b,
            vn,
            vt,
            ks.vSpace.weights,
            ks.gas.K,
            dt,
            ctr[i, ks.pSpace.ny].dy,
            -1,
        )
        a2face[i, ks.pSpace.ny+1].fw .= KitBase.global_frame(
            a2face[i, ks.pSpace.ny+1].fw,
            0.,
            1.,
        )
    end

    update!(ks, ctr, a1face, a2face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))
    # inflow
    for j = 1:ks.pSpace.ny
        @. ctr[0, j].prim[2:3] = 2.0 * ctr[1, j].prim[2:3] - ctr[2, j].prim[2:3]
        ctr[0, j].w .= prim_conserve(ctr[0, j].prim, ks.gas.γ)
        ctr[0, j].h .= KitBase.maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[0, j].prim)
        ctr[0, j].b = @. ctr[0, j].h * ks.gas.K / 2.0 / ctr[0, j].prim[end]
    end
    # outflow
    for j = 1:ks.pSpace.ny
        @. ctr[ks.pSpace.nx+1, j].prim[1:3] = 2.0 * ctr[ks.pSpace.nx, j].prim[1:3] - ctr[ks.pSpace.nx-1, j].prim[1:3]
        ctr[ks.pSpace.nx+1, j].prim[4] = ctr[ks.pSpace.nx+1, j].prim[1] / 2.0 / 0.25
        ctr[ks.pSpace.nx+1, j].w .= prim_conserve(ctr[ks.pSpace.nx+1, j].prim, ks.gas.γ)
        ctr[ks.pSpace.nx+1, j].h .= KitBase.maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[ks.pSpace.nx+1, j].prim)
        ctr[ks.pSpace.nx+1, j].b = @. ctr[ks.pSpace.nx+1, j].h * ks.gas.K / 2.0 / ctr[ks.pSpace.nx+1, j].prim[end]
    end
end

plot_contour(ks, ctr)

@save "sol.jld2" ks ctr