using Kinetic, Plots, LinearAlgebra, JLD2, Flux
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

###
# initialize kinetic solver
###

cd(@__DIR__)
@load "nn.jld2" nn

D = Dict{Symbol,Any}()
begin
    D[:matter] = "gas"
    D[:case] = "sod"
    D[:space] = "1d1f1v"
    D[:flux] = "kfvs"
    D[:collision] = "bgk"
    D[:nSpecies] = 1
    D[:interpOrder] = 2
    D[:limiter] = "vanleer"
    D[:boundary] = "fix"
    D[:cfl] = 0.3
    D[:maxTime] = 0.1

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
    D[:inK] = 0.0
    D[:omega] = 0.81
    D[:alphaRef] = 1.0
    D[:omegaRef] = 0.5
end

ks = SolverSet(D)
ctr, face = init_fvm(ks, ks.ps, :dynamic_array; structarray = true)
#=for i in eachindex(ctr)
    prim = [2.0 * rand(), 0.0, 1 / rand()]

    ctr[i].prim .= prim
    ctr[i].w .= prim_conserve(prim, ks.gas.γ)
    ctr[i].f .= maxwellian(ks.vs.u, prim)
end=#
for i in eachindex(face)
    face[i].fw .= 0.0
    face[i].ff .= 0.0
end

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ctr[1].w)
for iter = 1:nt
    println("iteration: $iter")

    reconstruct!(ks, ctr)

    #evolve!(ks, ctr, face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    for i in eachindex(face)
        w = (ctr[i-1].w .+ ctr[i].w) ./ 2
        prim = (ctr[i-1].prim .+ ctr[i].prim) ./ 2
        sw = (ctr[i].w .- ctr[i-1].w) / ks.ps.dx[i]
        tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)

        regime = nn([w; sw; tau]) |> onecold
        #regime = ifelse(iter < 15, 2, nn([w; sw; tau]) |> onecold)

        if regime == 1
            flux_gks!(
                face[i].fw,
                face[i].ff,
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
        elseif regime == 2
            flux_kfvs!(
                face[i].fw,
                face[i].ff,
                ctr[i-1].f,
                ctr[i].f,
                ks.vs.u,
                ks.vs.weights,
                dt,
                ctr[i-1].sf,
                ctr[i].sf,
            )
        end
    end
    
    KitBase.update!(ks, ctr, face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))

    t += dt
    #if t > ks.set.maxTime || maximum(res) < 5.e-7
    #    break
    #end
end

plot_line(ks, ctr)

# test
regime = zeros(Int, ks.ps.nx)
for i = 1:ks.ps.nx
    sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
    tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
    x, y = regime_data(ks, ctr[i].w, ctr[i].prim, sw, ctr[i].f)

    regime[i] = nn(x) |> onecold
end

plot!(ks.ps.x[1:ks.ps.nx], regime)

regime1 = zeros(Int, ks.ps.nx)
KnGLL = zeros(ks.ps.nx)
for i = 1:ks.ps.nx
    sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
    L = abs(ctr[i].w[1] / sw[1])
    ℓ = (1/ctr[i].prim[end])^ks.gas.ω / ctr[i].prim[1] * sqrt(ctr[i].prim[end]) * ks.gas.Kn

    KnGLL[i] = ℓ / L
    regime1[i] = ifelse(KnGLL[i] >= 0.05, 2, 1)
end

plot!(ks.ps.x[1:ks.ps.nx], regime1)

plot(ks.ps.x[1:ks.ps.nx], KnGLL)