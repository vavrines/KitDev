using Kinetic, Plots
using Kinetic.KitBase
using Base.Threads: @threads
using ProgressMeter: @showprogress

function step!(
    w::T1,
    prim::T1,
    h::T2,
    b::T2,
    fwL::T1,
    fhL::T2,
    fbL::T2,
    fwR::T1,
    fhR::T2,
    fbR::T2,
    u::T3,
    v::T3,
    weights::T3,
    K,
    γ,
    μᵣ,
    ω,
    Pr,
    Δs,
    dt,
    RES,
    AVG,
    collision = :bgk,
) where {
    T1<:AbstractArray{<:AbstractFloat,1},
    T2<:AbstractArray{<:AbstractFloat,2},
    T3<:AbstractArray{<:AbstractFloat,2},
}

    #--- store W^n and calculate shakhov term ---#
    w_old = deepcopy(w)

    if collision == :shakhov
        q = heat_flux(h, b, prim, u, v, weights)

        MH_old = maxwellian(u, v, prim)
        MB_old = MH_old .* K ./ (2.0 * prim[end])
        SH, SB = shakhov(u, v, MH_old, MB_old, q, prim, Pr, K)
    else
        SH = zero(h)
        SB = zero(b)
    end

    #--- update W^{n+1} ---#
    @. w += (fwL - fwR) / Δs
    prim .= conserve_prim(w, γ)

    #--- record residuals ---#
    @. RES += (w - w_old)^2
    @. AVG += abs(w)

    #--- calculate M^{n+1} and tau^{n+1} ---#
    MH = maxwellian(u, v, prim)
    MB = MH .* K ./ (2.0 * prim[end])
    MH .+= SH
    MB .+= SB
    τ = vhs_collision_time(prim, μᵣ, ω)

    #--- update distribution function ---#
    for j in axes(v, 2), i in axes(u, 1)
        h[i, j] =
            (h[i, j] + (fhL[i, j] - fhR[i, j]) / Δs + dt / τ * MH[i, j]) / (1.0 + dt / τ)
        b[i, j] =
            (b[i, j] + (fbL[i, j] - fbR[i, j]) / Δs + dt / τ * MB[i, j]) / (1.0 + dt / τ)
    end

end

function update(KS, ctr, face, dt, residual)
    sumRes = zero(ctr[1].w)
    sumAvg = zero(ctr[1].w)

    @inbounds @threads for i = 2:KS.pSpace.nx-1
        step!(
            ctr[i].w,
            ctr[i].prim,
            ctr[i].h,
            ctr[i].b,
            face[i].fw,
            face[i].fh,
            face[i].fb,
            face[i+1].fw,
            face[i+1].fh,
            face[i+1].fb,
            KS.vSpace.u,
            KS.vs.v,
            KS.vSpace.weights,
            KS.gas.K,
            KS.gas.γ,
            KS.gas.μᵣ,
            KS.gas.ω,
            KS.gas.Pr,
            KS.ps.dx[i],
            dt,
            sumRes,
            sumAvg,
            :bgk,
        )
    end

    for i in eachindex(residual)
        residual[i] = sqrt(sumRes[i] * KS.pSpace.nx) / (sumAvg[i] + 1.e-7)
    end

    return nothing
end

begin
    set = Setup(case = "layer", space = "2d2f2v", maxTime = 0.2)
    ps = PSpace1D(-1.0, 1.0, 100, 1)
    vs = VSpace2D(-5.0, 5.0, 28, -5.0, 5.0, 28)
    gas = Gas(Kn = 1e-3, K = 1.0)
    fw = function (x)
        prim = zeros(4)
        if x <= 0
            prim .= [1.0, 0.0, 1.0, 1.0]
        else
            prim .= [1.0, 0.0, -1.0, 2.0]
        end

        return prim_conserve(prim, ks.gas.γ)
    end
    ib = IB2F(fw, vs, gas)
    ks = SolverSet(set, ps, vs, gas, ib)
end

ctr, face = init_fvm(ks, ks.ps)

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ctr[1].w)

@showprogress for iter = 1:50
    reconstruct!(ks, ctr)

    @threads for i = 1:ks.ps.nx+1
        #=w = (ctr[i-1].w .+ ctr[i].w) ./ 2
        prim = (ctr[i-1].prim .+ ctr[i].prim) ./ 2
        sw = (ctr[i].w .- ctr[i-1].w) / ks.ps.dx[i]

        L = abs(ctr[i].w[1] / sw[1])
        ℓ = (1/prim[end])^ks.gas.ω / prim[1] * sqrt(prim[end]) * ks.gas.Kn
        KnGLL = ℓ / L
        isNS = ifelse(KnGLL > 0.05, false, true)=#

        #h = (ctr[i-1].h .+ ctr[i].h) ./ 2
        #x, y = regime_data(ks, w, prim, sw, h)
        #isNS = ifelse(onecold(y) == 1, true, false)

        @inbounds flux_kfvs!(
            face[i].fw,
            face[i].fh,
            face[i].fb,
            ctr[i-1].h .+ 0.5 .* ctr[i-1].sh .* ks.ps.dx[i-1],
            ctr[i-1].b .+ 0.5 .* ctr[i-1].sb .* ks.ps.dx[i-1],
            ctr[i].h .- 0.5 .* ctr[i].sh .* ks.ps.dx[i],
            ctr[i].b .- 0.5 .* ctr[i].sb .* ks.ps.dx[i],
            ks.vs.u,
            ks.vs.v,
            ks.vs.weights,
            dt,
            1.0,
            ctr[i-1].sh,
            ctr[i-1].sb,
            ctr[i].sh,
            ctr[i].sb,
        )
    end

    update(ks, ctr, face, dt, res)

    t += dt
end

sol = zeros(ks.ps.nx, 4)
for i in axes(sol, 1)
    sol[i, :] .= ctr[i].prim
    sol[i, end] = 1 / sol[i, end]
end

plot(ks.ps.x[1:ks.ps.nx], sol)

# distribution function
fc = (ctr[end÷2].h + ctr[end÷2+1].h) ./ 2
plot(ks.vs.v[1, :], fc[end÷2, :])

# regime
