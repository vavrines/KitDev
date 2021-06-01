using Kinetic, ProgressMeter, Plots, LinearAlgebra, JLD2

###
# initialize kinetic solver
###

cd(@__DIR__)
D = Dict{Symbol,Any}()
begin
    D[:matter] = "gas"
    D[:case] = "sod"
    D[:space] = "1d2f1v"
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
    D[:nu] = 100
    D[:vMeshType] = "rectangle"
    D[:nug] = 0

    D[:knudsen] = 1e-4
    D[:mach] = 0.0
    D[:prandtl] = 1.0
    D[:inK] = 2.0
    D[:omega] = 0.81
    D[:alphaRef] = 1.0
    D[:omegaRef] = 0.5
end

ks = SolverSet(D)
ctr, face = init_fvm(ks, ks.ps, :dynamic_array; structarray = true)
for i in eachindex(face)
    face[i].fw .= 0.0
    #face[i].ff .= 0.0
end

function flux_tmp!(
    fw::X,
    wL::Y,
    wR::Y,
    γ::Real,
    inK::Real,
    μᵣ::Real,
    ω::Real,
    dt::Real,
    dxL::Real,
    dxR::Real,
    swL::Y,
    swR::Y,
) where {X<:AbstractArray{<:AbstractFloat,1},Y<:AbstractArray{<:AbstractFloat,1}}

    primL = conserve_prim(wL .+ swL .* dxL, γ)
    primR = conserve_prim(wR .- swR .* dxR, γ)

    Mu1, Mxi1, MuL1, MuR1 = gauss_moments(primL, inK)
    Mu2, Mxi2, MuL2, MuR2 = gauss_moments(primR, inK)

    w =
        primL[1] .* moments_conserve(MuL1, Mxi1, 0, 0) .+
        primR[1] .* moments_conserve(MuR2, Mxi2, 0, 0)
    prim = conserve_prim(w, γ)
    tau =
        vhs_collision_time(prim, μᵣ, ω)

    faL = pdf_slope(primL, swL, inK)
    sw = -primL[1] .* moments_conserve_slope(faL, Mu1, Mxi1, 1)
    faTL = pdf_slope(primL, sw, inK)

    faR = pdf_slope(primR, swR, inK)
    sw = -primR[1] .* moments_conserve_slope(faR, Mu2, Mxi2, 1)
    faTR = pdf_slope(primR, sw, inK)

    Mu, Mxi, MuL, MuR = gauss_moments(prim, inK)
    sw0L = (w .- wL) ./ dxL
    sw0R = (wR .- w) ./ dxR
    gaL = pdf_slope(prim, sw0L, inK)
    gaR = pdf_slope(prim, sw0R, inK)
    sw =
        -prim[1] .* (
            moments_conserve_slope(gaL, MuL, Mxi, 1) .+
            moments_conserve_slope(gaR, MuR, Mxi, 1)
        )
    # ga = pdf_slope(prim, sw, inK)
    # sw = -prim[1] .* moments_conserve_slope(ga, Mu, Mxi, 1)
    gaT = pdf_slope(prim, sw, inK)

    # time-integration constants
    Mt = zeros(5)
    Mt[4] = tau * (1.0 - exp(-dt / tau))
    Mt[5] = -tau * dt * exp(-dt / tau) + tau * Mt[4]
    Mt[1] = dt - Mt[4]
    Mt[2] = -tau * Mt[1] + Mt[5]
    Mt[3] = 0.5 * dt^2 - tau * Mt[1]

    # flux related to central distribution
    Muv = moments_conserve(Mu, Mxi, 1, 0)
    MauL = moments_conserve_slope(gaL, MuL, Mxi, 2)
    MauR = moments_conserve_slope(gaR, MuR, Mxi, 2)
    # Mau = moments_conserve_slope(ga, MuR, Mxi, 2)
    MauT = moments_conserve_slope(gaT, Mu, Mxi, 1)

    fw .=
        Mt[1] .* prim[1] .* Muv .+ Mt[2] .* prim[1] .* (MauL .+ MauR) .+
        Mt[3] .* prim[1] .* MauT
    # fw .= Mt[1] .* prim[1] .* Muv .+ Mt[2] .* prim[1] .* Mau .+ Mt[3] .* prim[1] .* MauT

    # flux related to upwind distribution
    MuvL = moments_conserve(MuL1, Mxi1, 1, 0)
    MauL = moments_conserve_slope(faL, MuL1, Mxi1, 2)
    MauLT = moments_conserve_slope(faTL, MuL1, Mxi1, 1)

    MuvR = moments_conserve(MuR2, Mxi2, 1, 0)
    MauR = moments_conserve_slope(faR, MuR2, Mxi2, 2)
    MauRT = moments_conserve_slope(faTR, MuR2, Mxi2, 1)

    @. fw +=
        Mt[4] * primL[1] * MuvL - (Mt[5] + tau * Mt[4]) * primL[1] * MauL -
        tau * Mt[4] * primL[1] * MauLT + Mt[4] * primR[1] * MuvR -
        (Mt[5] + tau * Mt[4]) * primR[1] * MauR - tau * Mt[4] * primR[1] * MauRT
    # @. fw += Mt[4] * primL[1] * MuvL + Mt[4] * primR[1] * MuvR

    return nothing

end

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ks.ib.wL)
for iter = 1:nt
    reconstruct!(ks, ctr)

    for i in eachindex(face)
        flux_gks!(
            face[i].fw,
            ctr[i-1].w .+ ctr[i-1].sw .* ks.ps.dx[i-1] / 2,
            ctr[i].w .- ctr[i].sw .* ks.ps.dx[i] / 2,
            ks.gas.γ,
            ks.gas.K,
            ks.gas.μᵣ,
            ks.gas.ω,
            dt,
            ks.ps.dx[i-1] / 2,
            ks.ps.dx[i] / 2,
            ctr[i-1].sw,
            ctr[i].sw,
        )
        #=flux_hll!(
            face[i].fw,
            ctr[i-1].w,
            ctr[i].w,
            ks.gas.γ,
            dt,
        )=#
    end
    
    for i = 1:ks.ps.nx
        @. ctr[i].w += (face[i].fw - face[i+1].fw) / ks.ps.dx[i]
        ctr[i].prim .= conserve_prim(ctr[i].w, ks.gas.γ)
    end

    t += dt
    #if t > ks.set.maxTime || maximum(res) < 5.e-7
    #    break
    #end
end

plot_line(ks, ctr)

sol = zeros(ks.ps.nx, 3)
for i = 1:ks.ps.nx
    sol[i, :] .= ctr.prim[i]
    sol[i, 3] = 1 / sol[i, 3]
end

plot(ks.ps.x[1:ks.ps.nx], sol0[:, 3])
plot!(ks.ps.x[1:ks.ps.nx], sol[:, 3], line=:dash)
