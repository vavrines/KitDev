using KitBase, OrdinaryDiffEq, ProgressMeter

begin
    D = Dict{Symbol,Any}()

    D[:matter] = "gas"
    D[:case] = "sod"
    D[:space] = "1d1f3v"
    D[:flux] = "kfvs"
    D[:collision] = "bgk"
    D[:nSpecies] = 1
    D[:interpOrder] = 2
    D[:limiter] = "vanleer"
    D[:boundary] = "fix"
    D[:cfl] = 0.8
    D[:maxTime] = 0.2

    D[:x0] = 0.0
    D[:x1] = 1.0
    D[:nx] = 100
    D[:pMeshType] = "uniform"
    D[:nxg] = 1

    D[:umin] = -5.0
    D[:umax] = 5.0
    D[:nu] = 28
    D[:vmin] = -5.0
    D[:vmax] = 5.0
    D[:nv] = 16
    D[:wmin] = -5.0
    D[:wmax] = 5.0
    D[:nw] = 16
    D[:vMeshType] = "rectangle"
    D[:nug] = 0
    D[:nvg] = 0
    D[:nwg] = 0

    D[:knudsen] = 1.0
    D[:mach] = 0.0
    D[:prandtl] = 2 / 3
    D[:inK] = 0.0
    D[:omega] = 0.5
    D[:alphaRef] = 1.0
    D[:omegaRef] = 0.5
end

ks, ctr, face, t = initialize(D)

kn_bzm = hs_boltz_kn(ks.gas.μᵣ, ks.gas.αᵣ)
phi, psi, phipsi = kernel_mode(
    5,
    ks.vSpace.u1,
    ks.vSpace.v1,
    ks.vSpace.w1,
    ks.vSpace.du[1, 1, 1],
    ks.vSpace.dv[1, 1, 1],
    ks.vSpace.dw[1, 1, 1],
    ks.vSpace.nu,
    ks.vSpace.nv,
    ks.vSpace.nw,
    ks.gas.αᵣ,
)

dt = timestep(ks, ctr, t)
nt = floor(ks.set.maxTime / dt) |> Int

function flux_bz!(
    fw::X,
    ff::Y,
    wL::Z,
    fL::A,
    wR::Z,
    fR::A,
    uVelo::B,
    vVelo::B,
    wVelo::B,
    ω::B,
    inK,
    γ,
    kn,
    nm,
    phi,
    psi,
    phipsi,
    dt,
) where {
    X<:AbstractArray{<:AbstractFloat,1},
    Y<:AbstractArray{<:AbstractFloat,3},
    Z<:AbstractArray{<:Real,1},
    A<:AbstractArray{<:AbstractFloat,3},
    B<:AbstractArray{<:AbstractFloat,3},
}

    # upwind reconstruction
    δ = heaviside.(uVelo)
    f = @. fL * δ + fR * (1.0 - δ)

    primL = conserve_prim(wL, γ)
    primR = conserve_prim(wR, γ)

    # construct interface distribution
    Mu1, Mv1, Mw1, MuL1, MuR1 = gauss_moments(primL, inK)
    Muv1 = moments_conserve(MuL1, Mv1, Mw1, 0, 0, 0)
    Mu2, Mv2, Mw2, MuL2, MuR2 = gauss_moments(primR, inK)
    Muv2 = moments_conserve(MuR2, Mv2, Mw2, 0, 0, 0)

    w = similar(wL)
    @. w = primL[1] * Muv1 + primR[1] * Muv2

    prim = conserve_prim(w, γ)

    prob = ODEProblem(boltzmann_ode!, f, (0.0, 0.5 * dt), [kn, nm, phi, psi, phipsi])
    sol = solve(prob, Midpoint(), saveat = 0.5 * dt)
    f .= sol.u[end]

    fw[1] = dt * sum(ω .* uVelo .* f)
    fw[2] = dt * sum(ω .* uVelo .^ 2 .* f)
    fw[3] = dt * sum(ω .* uVelo .* vVelo .* f)
    fw[4] = dt * sum(ω .* uVelo .* wVelo .* f)
    fw[5] = dt * 0.5 * sum(ω .* uVelo .* (uVelo .^ 2 .+ vVelo .^ 2 .+ wVelo .^ 2) .* f)

    @. ff = dt * uVelo * f

    return nothing

end

function evolve(KS::SolverSet, ctr::T1, face::T2, dt, p) where {T1,T2}

    if firstindex(KS.pSpace.x) < 1
        idx0 = 1
        idx1 = KS.pSpace.nx + 1
    else
        idx0 = 2
        idx1 = KS.pSpace.nx
    end

    kn, nm, phi, psi, phipsi = p

    @inbounds Threads.@threads for i = idx0:idx1
        flux_bz!(
            face[i].fw,
            face[i].ff,
            ctr[i-1].w,
            ctr[i-1].f,
            ctr[i].w,
            ctr[i].f,
            KS.vSpace.u,
            KS.vSpace.v,
            KS.vSpace.w,
            KS.vSpace.weights,
            KS.gas.K,
            KS.gas.γ,
            kn,
            nm,
            phi,
            psi,
            phipsi,
            dt,
        )
    end

end

function step_bz!(
    fwL::T1,
    ffL::T2,
    w::T3,
    prim::T3,
    f::T4,
    fwR::T1,
    ffR::T2,
    uVelo::T5,
    vVelo::T5,
    wVelo::T5, # avoid conflict with w
    weights::T5,
    K,
    γ,
    kn,
    Pr,
    nm,
    phi,
    psi,
    phipsi,
    dx,
    dt,
    RES,
    AVG,
    collision = :bgk::Symbol,
) where {
    T1<:AbstractArray{<:AbstractFloat,1},
    T2<:AbstractArray{<:AbstractFloat,3},
    T3<:AbstractArray{<:AbstractFloat,1},
    T4<:AbstractArray{<:AbstractFloat,3},
    T5<:AbstractArray{<:AbstractFloat,3},
}

    #--- store W^n and calculate shakhov term ---#
    w_old = deepcopy(w)

    #q = heat_flux(f, prim, uVelo, vVelo, wVelo, weights)
    #M_old = maxwellian(uVelo, vVelo, wVelo, prim)
    #S = shakhov(uVelo, vVelo, wVelo, M_old, q, prim, Pr, K)

    #--- update W^{n+1} ---#
    @. w += (fwL - fwR) / dx
    prim .= conserve_prim(w, γ)

    #--- record residuals ---#
    @. RES += (w - w_old)^2
    @. AVG += abs(w)

    #--- calculate M^{n+1} and tau^{n+1} ---#
    #M = maxwellian(uVelo, vVelo, wVelo, prim)
    #M .+= S
    #τ = vhs_collision_time(prim, μᵣ, ω)

    df = similar(f)
    boltzmann_ode!(df, f, (kn, nm, phi, psi, phipsi), dt)

    #--- update distribution function ---#
    for k in axes(wVelo, 3), j in axes(vVelo, 2), i in axes(uVelo, 1)
        f[i, j, k] += (ffL[i, j, k] - ffR[i, j, k]) / dx + dt * df[i, j, k]
    end

end

function update(
    KS::X,
    ctr::Y,
    face::Z,
    dt,
    residual,
    p,
) where {
    X<:AbstractSolverSet,
    Y<:AbstractArray{ControlVolume1D1F,1},
    Z<:AbstractArray{Interface1D1F,1},
}

    sumRes = zeros(axes(KS.ib.wL))
    sumAvg = zeros(axes(KS.ib.wL))

    kn, nm, phi, psi, phipsi = p

    @inbounds Threads.@threads for i = 2:KS.pSpace.nx-1
        step_bz!(
            face[i].fw,
            face[i].ff,
            ctr[i].w,
            ctr[i].prim,
            ctr[i].f,
            face[i+1].fw,
            face[i+1].ff,
            KS.vSpace.u,
            KS.vSpace.v,
            KS.vSpace.w,
            KS.vSpace.weights,
            KS.gas.K,
            KS.gas.γ,
            KS.gas.Pr,
            kn,
            nm,
            phi,
            psi,
            phipsi,
            ctr[i].dx,
            dt,
            sumRes,
            sumAvg,
            :boltzmann,
        )
    end

    for i in eachindex(residual)
        residual[i] = sqrt(sumRes[i] * KS.pSpace.nx) / (sumAvg[i] + 1.e-7)
    end

    return nothing

end


res = zeros(5)
@showprogress for iter = 1:nt÷10
    evolve(ks, ctr, face, dt, (kn_bzm, 5, phi, psi, phipsi))
    update(ks, ctr, face, dt, res, (kn_bzm, 5, phi, psi, phipsi))

    #iter += 1
    t += dt
end

plot_line(ks, ctr)
