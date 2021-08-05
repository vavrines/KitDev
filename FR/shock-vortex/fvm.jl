using KitBase, Plots, OffsetArrays

begin
    set = Setup(
        "gas",
        "cylinder",
        "2d0f",
        "hll",
        "nothing",
        1, # species
        3, # order of accuracy
        "positivity", # limiter
        "euler",
        0.1, # cfl
        1.0, # time
    )
    ps = PSpace2D(0.0, 2.0, 100, 0.0, 1.0, 50, 1, 1)
    vs = VSpace2D(-2.0, 2.0, 5, -2.0, 2.0, 5)
    gas = Gas(
        1e-6,
        1.12, # Mach
        1.0,
        3.0, # K
        7/5,
        0.81,
        1.0,
        0.5,
    )
    ib = nothing
    ks = SolverSet(set, ps, vs, gas, ib)
end

ctr = OffsetArray{ControlVolume2D}(undef, axes(ks.ps.x, 1), axes(ks.ps.y, 2))
a1face = Array{Interface2D}(undef, ks.ps.nx + 1, ks.ps.ny)
a2face = Array{Interface2D}(undef, ks.ps.nx, ks.ps.ny + 1)

for j in axes(ctr, 2), i in axes(ctr, 1)
    t1 = ib_rh(ks.gas.Ma, ks.gas.γ, rand(3))[2]
    t2 = ib_rh(ks.gas.Ma, ks.gas.γ, rand(3))[6]

    if i <= ks.ps.nx ÷ 2
        prim = [t2[1], t1[2] - t2[2], 0.0, t2[3]]
    else
        prim = [t1[1], 0.0, 0.0, t1[3]]
    end
    w = prim_conserve(prim, ks.gas.γ)

    ctr[i, j] = ControlVolume2D(
        ks.ps.x[i, j],
        ks.ps.y[i, j],
        ks.ps.dx[i, j],
        ks.ps.dy[i, j],
        w,
        prim,
    )
end

for j = 1:ks.ps.ny
    for i = 1:ks.ps.nx+1
        a1face[i, j] = Interface2D(ks.ps.dy[i, j], 1.0, 0.0, zeros(4))
    end
end
for i = 1:ks.ps.nx
    for j = 1:ks.ps.ny+1
        a2face[i, j] =
            Interface2D(ks.ps.dx[i, j], 0.0, 1.0, zeros(4))
    end
end

iter = 0
t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(floor(ks.set.maxTime / dt)) + 1
res = zero(4)

reconstruct!(ks, ctr)
evolve!(ks, ctr, a1face, a2face, dt)
update!(ks, ctr, a1face, a2face, dt, res)

plot_contour(ks, ctr)

