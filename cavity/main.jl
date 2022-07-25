using Kinetic
cd(@__DIR__)
ks, ctr, a1face, a2face, t = initialize("config.txt")
t = solve!(ks, ctr, a1face, a2face, t)

using ProgressMeter
res = zeros(4)
dt = timestep(ks, ctr, t)
nt = floor(ks.set.maxTime / dt) |> Int
@showprogress for iter = 1:nt
    #reconstruct!(ks, ctr)
    evolve!(
        ks,
        ctr,
        a1face,
        a2face,
        dt;
        mode = Symbol(ks.set.flux),
        bc = Symbol(ks.set.boundary),
    )
    Kinetic.update!(
        ks,
        ctr,
        a1face,
        a2face,
        dt,
        res;
        coll = Symbol(ks.set.collision),
        bc = Symbol(ks.set.boundary),
    )
end

plot_contour(ks, ctr)
