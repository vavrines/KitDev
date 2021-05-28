using Kinetic, ProgressMeter, Plots, LinearAlgebra

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
    L = norm((ctr[i].f .- fr) ./ prim[1])

    x = [w; sw; tau]
    y = ifelse(L <= 0.05, [1.0, 0.0], [0.0, 1.0])
    return x, y
end

@showprogress for iter = 1:5000#nt
    if iter%39 == 0
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
plot(ks.vs.u, ctr[i].f)
plot!(ks.vs.u, fr, line=:dash)

using Flux

nn = Chain(
    Dense(7, 14, tanh),
    Dense(14, 28, tanh),
    Dense(28, 14, tanh),
    Dense(14, 2),
)

sci_train!(nn, (X, Y), ADAM(); device = cpu, epoch = 50)

# test
i = 50
sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
x, y = regime_data(ks, ctr[i].w, ctr[i].prim, sw, ctr[i].f)
nn(x)

# accuracy
nn(X)
Y
