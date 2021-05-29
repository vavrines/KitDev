using Kinetic, ProgressMeter, Plots, LinearAlgebra

###
# initialize kinetic solver
###

cd(@__DIR__)
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
    D[:boundary] = "period"
    D[:cfl] = 0.5
    D[:maxTime] = 5.0

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

    D[:knudsen] = 0.05
    D[:mach] = 0.0
    D[:prandtl] = 1.0
    D[:inK] = 0.0
    D[:omega] = 0.81
    D[:alphaRef] = 1.0
    D[:omegaRef] = 0.5
end

ks = SolverSet(D)
ctr, face = init_fvm(ks, ks.ps, :dynamic_array; structarray = true)
for i in eachindex(ctr)
    prim = [2.0 * rand(), 0.0, 1 / rand()]

    ctr[i].prim .= prim
    ctr[i].w .= prim_conserve(prim, ks.gas.γ)
    ctr[i].f .= maxwellian(ks.vs.u, prim)
end
for i in eachindex(face)
    face[i].fw .= 0.0
    face[i].ff .= 0.0
end

plot_line(ks, ctr)

###
# prepare training set
###

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ks.ib.wL)

X = Float32.([[1e-4, 0.0, 1e-4]; zeros(3); 1.0])
Y = Float32.([1.0, 0.0])

function regime_data(ks, w, prim, sw, f)
    Mu, Mxi, _, _1 = gauss_moments(prim, ks.gas.K)
    a = pdf_slope(prim, sw, ks.gas.K)
    sw = -prim[1] .* moments_conserve_slope(a, Mu, Mxi, 1)
    A = pdf_slope(prim, sw, ks.gas.K)
    tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
    fr = chapman_enskog(ks.vs.u, prim, a, A, tau)
    L = norm((f .- fr) ./ prim[1])

    x = [w; sw; tau]
    y = ifelse(L <= 0.01, [1.0, 0.0], [0.0, 1.0])
    return x, y
end

@showprogress for iter = 1:3000#nt
    if iter%13 == 0
        for i = 1:ks.ps.nx
            sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
            x, y = regime_data(ks, ctr[i].w, ctr[i].prim, sw, ctr[i].f)
            X = hcat(X, x)
            Y = hcat(Y, y)
        end
    end
    
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    KitBase.update!(ks, ctr, face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))

    t += dt
    #if t > ks.set.maxTime || maximum(res) < 5.e-7
    #    break
    #end
end

plot_line(ks, ctr)

###
# define neural model
###

using Flux
using Flux: onecold

nn = Chain(
    Dense(7, 14, relu),
    Dense(14, 28, relu),
    Dense(28, 14, relu),
    Dense(14, 2),
)

sci_train!(nn, (X, Y), ADAM(); device = cpu, epoch = 50)

# test
i = 50
sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
x, y = regime_data(ks, ctr[i].w, ctr[i].prim, sw, ctr[i].f)
nn(x) #|> onecold

Mu, Mxi, _, _1 = gauss_moments(ctr[i].prim, ks.gas.K)
a = pdf_slope(ctr[i].prim, sw, ks.gas.K)
sw = -ctr[i].prim[1] .* moments_conserve_slope(a, Mu, Mxi, 1)
A = pdf_slope(ctr[i].prim, sw, ks.gas.K)
fr = chapman_enskog(ks.vs.u, ctr[i].prim, a, A, tau)

plot(ks.vs.u, ctr[i].f)
plot!(ks.vs.u, fr, line=:dash)

# accuracy
Y1 = nn(X)
Y

YA1 = [onecold(Y1[:, i]) for i in axes(Y1, 2)]
YA = [onecold(Y[:, i]) for i in axes(Y, 2)]

accuracy = 0.0
for i in eachindex(YA)
    if YA[i] == YA1[i]
        accuracy += 1.0
    end
end
accuracy /= length(YA)

###
# hybrid solver
###

begin
    for i in eachindex(ctr)
        prim = [2.0 * rand(), 0.0, 1 / rand()]

        ctr[i].prim .= prim
        ctr[i].w .= prim_conserve(prim, ks.gas.γ)
        ctr[i].f .= maxwellian(ks.vs.u, prim)
    end
    for i in eachindex(face)
        face[i].fw .= 0.0
        face[i].ff .= 0.0
    end
end

for iter = 1:1000#nt
    println("iteration: $iter")

    reconstruct!(ks, ctr)

    #evolve!(ks, ctr, face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    for i in eachindex(face)
        w = (ctr[i-1].w .+ ctr[i].w) ./ 2
        prim = (ctr[i-1].prim .+ ctr[i].prim) ./ 2
        sw = (ctr[i].w .- ctr[i-1].w) / ks.ps.dx[i]
        tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)

        #regime = nn([w; sw; tau]) |> onecold
        regime = ifelse(iter < 15, 2, nn([w; sw; tau]) |> onecold)

        if regime == 1
            flux_gks!(
                face[i].fw,
                face[i].ff,
                ctr[i-1].w,
                ctr[i].w,
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
