using Kinetic, ProgressMeter, Plots, LinearAlgebra, JLD2

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

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(floor(ks.set.maxTime / dt))
res = zeros(3)
for iter = 1:nt
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, face, dt)
    Kinetic.update!(ks, ctr, face, dt, res)
end

plot_line(ks, ctr)

sol0 = zeros(ks.ps.nx, 3)
for i = 1:ks.ps.nx
    sol0[i, :] .= ctr.prim[i]
    sol0[i, 3] = 1 / sol0[i, 3]
end
