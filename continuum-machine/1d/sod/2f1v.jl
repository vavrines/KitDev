using Kinetic, Plots, LinearAlgebra, Flux, JLD2
using Flux: onecold
using ProgressMeter: @showprogress

function regime_data(ks, w, prim, sw, f)
    Mu, Mxi, _, _1 = gauss_moments(prim, ks.gas.K)
    a = pdf_slope(prim, sw, ks.gas.K)
    sw = -prim[1] .* moments_conserve_slope(a, Mu, Mxi, 1)
    A = pdf_slope(prim, sw, ks.gas.K)
    tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
    fr = chapman_enskog(ks.vs.u, prim, a, A, tau)
    L = norm((f .- fr) ./ prim[1])

    x = [w; sw; tau]
    y = ifelse(L <= 0.005, [1.0, 0.0], [0.0, 1.0])
    return x, y
end

set = Setup(case = "sod", space = "1d2f1v", maxTime = 0.15)
ps = PSpace1D(0.0, 1.0, 200, 1)
vs = VSpace1D(-5.0, 5.0, 100)
gas = Gas(Kn = 1e-4)
ib = IB2F(ib_sod(set, ps, vs, gas)...)
ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks, ks.ps)

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ctr[1].w)

@showprogress for iter = 1:nt
    reconstruct!(ks, ctr)

    for i = 1:ks.ps.nx+1
        w = (ctr[i-1].w .+ ctr[i].w) ./ 2
        prim = (ctr[i-1].prim .+ ctr[i].prim) ./ 2
        sw = (ctr[i].w .- ctr[i-1].w) / ks.ps.dx[i]
        
        L = abs(ctr[i].w[1] / sw[1])
        ℓ = (1/prim[end])^ks.gas.ω / prim[1] * sqrt(prim[end]) * ks.gas.Kn
        KnGLL = ℓ / L
        isNS = ifelse(KnGLL > 0.05, false, true)

        #h = (ctr[i-1].h .+ ctr[i].h) ./ 2
        #x, y = regime_data(ks, w, prim, sw, h)
        #isNS = ifelse(onecold(y) == 1, true, false)

        if isNS
            flux_gks!(
                face[i].fw,
                face[i].fh,
                face[i].fb,
                ctr[i-1].w .+ ctr[i-1].sw .* ks.ps.dx[i-1] / 2,
                ctr[i].w .- ctr[i].sw .* ks.ps.dx[i] / 2,
                ks.vs.u,
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                dt,
                ks.ps.dx[i-1] / 2,
                ks.ps.dx[i] / 2,
                ctr[i-1].sw,
                ctr[i].sw,
            )
        else
            flux_kfvs!(
                face[i].fw,
                face[i].fh,
                face[i].fb,
                ctr[i-1].h .+ 0.5 .* ctr[i-1].sh .* ks.ps.dx[i-1],
                ctr[i-1].b .+ 0.5 .* ctr[i-1].sb .* ks.ps.dx[i-1],
                ctr[i].h .- 0.5 .* ctr[i].sh .* ks.ps.dx[i],
                ctr[i].b .- 0.5 .* ctr[i].sb .* ks.ps.dx[i],
                ks.vs.u,
                ks.vs.weights,
                dt,
                ctr[i-1].sh,
                ctr[i-1].sb,
                ctr[i].sh,
                ctr[i].sb,
            )
        end
    end
    
    update!(ks, ctr, face, dt, res; coll = :bgk, bc = [:fix, :fix])

    t += dt
end

plot(ks, ctr)
plot!(ks, ctr0)

cd(@__DIR__)
#@save "pure_kinetic.jld2" ks ctr
#@save "pure_ns.jld2" ks ctr
#@save "nn.jld2" ks ctr
#@save "kngll.jld2" ks ctr

begin
    regime = zeros(Int, ks.ps.nx)
    for i = 1:ks.ps.nx
        sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
        x, y = regime_data(ks, ctr[i].w, ctr[i].prim, sw, ctr[i].h)
        regime[i] = onecold(y)
    end

    regime1 = zeros(Int, ks.ps.nx)
    KnGLL = zeros(ks.ps.nx)
    for i = 1:ks.ps.nx
        sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
        L = abs(ctr[i].w[1] / sw[1])
        ℓ = (1/ctr[i].prim[end])^ks.gas.ω / ctr[i].prim[1] * sqrt(ctr[i].prim[end]) * ks.gas.Kn

        KnGLL[i] = ℓ / L
        regime1[i] = ifelse(KnGLL[i] >= 0.05, 2, 1)
    end
end

plot(ks.ps.x[1:ks.ps.nx], regime)
plot!(ks.ps.x[1:ks.ps.nx], regime1)
