mutable struct CellInfo
    regime::String
    ispdf::Bool
end

function recon_pdf(ks, prim, swx, swy)
    Mu, Mv, Mxi, _, _1 = gauss_moments(prim, ks.gas.K)
    a = pdf_slope(prim, swx, ks.gas.K)
    b = pdf_slope(prim, swy, ks.gas.K)
    sw =
        -prim[1] .* (
            moments_conserve_slope(a, Mu, Mv, Mxi, 1, 0) .+
            moments_conserve_slope(b, Mu, Mv, Mxi, 0, 1)
        )
    A = pdf_slope(prim, sw, ks.gas.K)
    tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)

    return chapman_enskog(ks.vs.u, ks.vs.v, prim, a, b, A, tau)
end

function judge_regime(f, fr, prim)
    L = norm((f .- fr) ./ prim[1])
    return ifelse(L <= 0.005, 1, 2)
end

function judge_regime(ks, f, prim, swx, swy)
    fr = recon_pdf(ks, prim, swx, swy)
    return judge_regime(f, fr, prim)
end

function rc!(
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
        reconstruct2!(swR, ctr[nx-1, j].w, ctr[nx, j].w, 0.5 * (dx[nx-1, j] + dx[nx, j]))
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
        reconstruct2!(shR, ctr[nx-1, j].h, ctr[nx, j].h, 0.5 * (dx[nx-1, j] + dx[nx, j]))
        sbR = KitBase.extract_last(ctr[nx, j].sb, 1, mode = :view)
        reconstruct2!(sbR, ctr[nx-1, j].b, ctr[nx, j].b, 0.5 * (dx[nx-1, j] + dx[nx, j]))
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
