using KitBase, Plots, OffsetArrays
using ProgressMeter: @showprogress

begin
    set = Setup(
        "gas",
        "cylinder",
        "2d0f",
        "hll",
        "nothing",
        1, # species
        3, # order of accuracy
        "vanleer", # limiter
        "extra",
        0.5, # cfl
        1.0, # time
    )
    ps = PSpace2D(0.0, 2.0, 150, 0.0, 1.0, 75, 1, 1)
    vs = VSpace2D(-2.0, 2.0, 5, -2.0, 2.0, 5)
    gas = Gas(
        1e-6,
        1.1381581917106134, # Mach
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

begin
    gam = gas.γ
    MaL = gas.Ma
    MaR = sqrt((MaL^2 * (gam - 1.0) + 2.0) / (2.0 * gam * MaL^2 - (gam - 1.0)))
    ratioT =
        (1.0 + (gam - 1.0) / 2.0 * MaL^2) * (2.0 * gam / (gam - 1.0) * MaL^2 - 1.0) /
        (MaL^2 * (2.0 * gam / (gam - 1.0) + (gam - 1.0) / 2.0))
    t1 = [1.0, MaL * sqrt(gam / 2.0), 0.0, 1.0]
    t2 = [
        t1[1] * (gam + 1.0) * MaL^2 / ((gam - 1.0) * MaL^2 + 2.0),
        MaR * sqrt(gam / 2.0) * sqrt(ratioT),
        0.0,
        t1[end] / ratioT,
    ]
end

for j in axes(ctr, 2), i in axes(ctr, 1)
    if ps.x[i, j] <= ps.x1*0.25
        prim = [t2[1], t1[2] - t2[2], 0.0, t2[end]]
    else
        prim = [t1[1], 0.0, 0.0, t1[end]]

        s = prim[1]^(1-ks.gas.γ) / (2 * prim[end])

        κ = 0.25 # vortex strength
        μ = 0.204
        rc = 0.05

        x0 = 0.8
        y0 = 0.5

        r = sqrt((ps.x[i, j] - x0)^2 + (ps.y[i, j] - y0)^2)
        
        
        η = r / rc
        
        δu = κ * η * exp(μ * (1-η^2)) * (ps.y[i, j] - y0) / r
        δv = -κ * η * exp(μ * (1-η^2)) * (ps.x[i, j] - x0) / r
        δT = -(ks.gas.γ-1)*κ^2/(8*μ*ks.gas.γ)*exp(2*μ*(1-η^2))

        T0 = 1 / prim[end]

        ρ = prim[1]^(ks.gas.γ-1) * (T0+δT) / T0^(1/(ks.gas.γ-1))
        #@show prim[1]

        #@show (1 / (s * 2 * (prim[end] + δλ)))

        #ρ = (1 / (s * 2 * (prim[end] + δλ)))^(1/(ks.gas.γ-1))
        prim1 = [ρ, prim[2]+δu, prim[3]+δv, 1/(1/prim[4]+δT)]

        if r <= rc * 8
            prim .= prim1
        end
    end
    w = prim_conserve(prim, ks.gas.γ)

    ctr[i, j] = ControlVolume2D(
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
dt = KitBase.timestep(ks, ctr, t)
nt = Int(floor(ks.set.maxTime / dt)) + 1
res = zero(4)

@showprogress for iter = 1:nt
    KitBase.reconstruct!(ks, ctr)
    KitBase.evolve!(ks, ctr, a1face, a2face, dt)
    #update!(ks, ctr, a1face, a2face, dt, res)

    @inbounds for j = 1:ks.ps.ny
        for i = 1:ks.ps.nx
            KitBase.step!(
                ctr[i, j].w,
                ctr[i, j].prim,
                a1face[i, j].fw,
                a1face[i+1, j].fw,
                a2face[i, j].fw,
                a2face[i, j+1].fw,
                ks.gas.γ,
                ks.ps.dx[i, j] * ks.ps.dy[i, j],
                zeros(4),
                zeros(4),
            )
        end
    end
    for i = 1:ks.ps.nx
        ctr[i, 0].w .= ctr[i, 1].w
        ctr[i, 0].prim .= ctr[i, 1].prim
        ctr[i, ks.ps.ny+1].w .= ctr[i, ks.ps.ny].w
        ctr[i, ks.ps.ny+1].prim .= ctr[i, ks.ps.ny].prim
    end
    for j = 1:ks.ps.ny
        ctr[ks.ps.nx+1, j].w .= ctr[ks.ps.nx, j].w
        ctr[ks.ps.nx+1, j].prim .= ctr[ks.ps.nx, j].prim
    end
end

begin
    sol = zeros(ps.nx, ps.ny, 4)
    for i = 1:ps.nx, j = 1:ps.ny
        sol[i, j, :] .= conserve_prim(ctr[i, j].w, ks.gas.γ)
        sol[i, j, 4] = 1 / sol[i, j, 4]
    end
    x = ps.x[1:ps.nx, 1]
    plot(x, 0.5 .* (sol[:, end÷2, 1] + sol[:, end÷2+1, 1]))
end

#plot_contour(ks, ctr)

using JLD2
cd(@__DIR__)
@save "fvm6.jld2" x sol ctr