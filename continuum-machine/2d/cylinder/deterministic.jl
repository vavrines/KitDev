using KitBase, Plots, JLD2
using KitBase.ProgressMeter: @showprogress
pyplot()

function rcnew!(
    KS::X,
    ctr::Y,
) where {X<:AbstractSolverSet,Y<:AbstractArray{ControlVolume2D2F,2}}

    if KS.set.interpOrder == 1
        return
    end

    nx, ny, dx, dy = begin
        if KS.ps isa CSpace2D
            KS.ps.nr, KS.ps.nθ, KS.ps.dr, KS.ps.darc
        else
            KS.ps.nx, KS.ps.ny, KS.ps.dx, KS.ps.dy
        end
    end

    #--- conservative variables ---#
    # boundary
    @inbounds Threads.@threads for j = 1:ny
        #swL = extract_last(ctr[1, j].sw, 1, mode = :view)
        #reconstruct2!(swL, ctr[1, j].w, ctr[2, j].w, 0.5 * (dx[1, j] + dx[2, j]))

        swR = KitBase.extract_last(ctr[nx, j].sw, 1, mode = :view)
        reconstruct2!(
            swR,
            ctr[nx-1, j].w,
            ctr[nx, j].w,
            0.5 * (dx[nx-1, j] + dx[nx, j]),
        )
    end

    # inner
    @inbounds Threads.@threads for j = 1:ny
        for i = 2:nx-1
            sw = KitBase.extract_last(ctr[i, j].sw, 1, mode = :view)
            reconstruct3!(
                sw,
                ctr[i-1, j].w,
                ctr[i, j].w,
                ctr[i+1, j].w,
                0.5 * (dx[i-1, j] + dx[i, j]),
                0.5 * (dx[i, j] + dx[i+1, j]),
                Symbol(KS.set.limiter),
            )
        end
    end

    @inbounds Threads.@threads for j = 1:ny
        for i = 1:nx
            sw = KitBase.extract_last(ctr[i, j].sw, 2, mode = :view)
            reconstruct3!(
                sw,
                ctr[i, j-1].w,
                ctr[i, j].w,
                ctr[i, j+1].w,
                0.5 * (dy[i, j-1] + dy[i, j]),
                0.5 * (dy[i, j] + dy[i, j+1]),
                Symbol(KS.set.limiter),
            )
        end
    end

    #--- particle distribution function ---#
    # boundary
    @inbounds Threads.@threads for j = 1:ny
        #shL = extract_last(ctr[1, j].sh, 1, mode = :view)
        #reconstruct2!(shL, ctr[1, j].h, ctr[2, j].h, 0.5 * (dx[1, j] + dx[2, j]))
        #sbL = extract_last(ctr[1, j].sb, 1, mode = :view)
        #reconstruct2!(sbL, ctr[1, j].b, ctr[2, j].b, 0.5 * (dx[1, j] + dx[2, j]))

        shR = KitBase.extract_last(ctr[nx, j].sh, 1, mode = :view)
        reconstruct2!(
            shR,
            ctr[nx-1, j].h,
            ctr[nx, j].h,
            0.5 * (dx[nx-1, j] + dx[nx, j]),
        )
        sbR = KitBase.extract_last(ctr[nx, j].sb, 1, mode = :view)
        reconstruct2!(
            sbR,
            ctr[nx-1, j].b,
            ctr[nx, j].b,
            0.5 * (dx[nx-1, j] + dx[nx, j]),
        )
    end

    # inner
    @inbounds Threads.@threads for j = 1:ny
        for i = 2:nx-1
            sh = KitBase.extract_last(ctr[i, j].sh, 1, mode = :view)
            reconstruct3!(
                sh,
                ctr[i-1, j].h,
                ctr[i, j].h,
                ctr[i+1, j].h,
                0.5 * (dx[i-1, j] + dx[i, j]),
                0.5 * (dx[i, j] + dx[i+1, j]),
                Symbol(KS.set.limiter),
            )

            sb = KitBase.extract_last(ctr[i, j].sb, 1, mode = :view)
            reconstruct3!(
                sb,
                ctr[i-1, j].b,
                ctr[i, j].b,
                ctr[i+1, j].b,
                0.5 * (dx[i-1, j] + dx[i, j]),
                0.5 * (dx[i, j] + dx[i+1, j]),
                Symbol(KS.set.limiter),
            )
        end
    end

    @inbounds Threads.@threads for j = 1:ny
        for i = 1:nx
            sh = KitBase.extract_last(ctr[i, j].sh, 2, mode = :view)
            reconstruct3!(
                sh,
                ctr[i, j-1].h,
                ctr[i, j].h,
                ctr[i, j+1].h,
                0.5 * (dy[i, j-1] + dy[i, j]),
                0.5 * (dy[i, j] + dy[i, j+1]),
                Symbol(KS.set.limiter),
            )

            sb = KitBase.extract_last(ctr[i, j].sb, 2, mode = :view)
            reconstruct3!(
                sb,
                ctr[i, j-1].b,
                ctr[i, j].b,
                ctr[i, j+1].b,
                0.5 * (dy[i, j-1] + dy[i, j]),
                0.5 * (dy[i, j] + dy[i, j+1]),
                Symbol(KS.set.limiter),
            )
        end
    end

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
    gas = Gas(Kn = 5e-2, Ma = 5.0, K = 1.0)
    
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
cd(@__DIR__)
#@load "restart.jld2" ctr

t = 0.0
dt = timestep(ks, ctr, 0.0)
nt = ks.set.maxTime ÷ dt |> Int
@showprogress for iter = 1:100#nt
    #reconstruct!(ks, ctr)
    rcnew!(ks, ctr)
    evolve!(ks, ctr, a1face, a2face, dt)
    update!(ks, ctr, a1face, a2face, dt, zeros(4))

    for j = ks.ps.nθ÷2+1:ks.ps.nθ
        ctr[ks.ps.nr+1, j].w .= ks.ib.fw(6, 0)
        ctr[ks.ps.nr+1, j].prim .= conserve_prim(ctr[ks.ps.nr+1, j].w, ks.gas.γ)
        ctr[ks.ps.nr+1, j].sw .= 0.0
        ctr[ks.ps.nr+1, j].h .= maxwellian(ks.vs.u, ks.vs.v, ctr[ks.ps.nr+1, j].prim)
        ctr[ks.ps.nr+1, j].b .= ctr[ks.ps.nr+1, j].h .* ks.gas.K ./ 2 ./ ctr[ks.ps.nr+1, j].prim[end]
    end

    global t += dt
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

#cd(@__DIR__)
#@save "restart.jld2" ctr
