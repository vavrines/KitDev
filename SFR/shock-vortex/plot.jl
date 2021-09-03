using Plots, JLD2, Langevin, FluxReconstruction

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
    ps = FRPSpace2D(0.0, 2.0, 100, 0.0, 1.0, 50, set.interpOrder-1, 1, 1)
    vs = nothing
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

    uqMethod = "galerkin"
    nr = 5
    nRec = 10
    opType = "uniform"
    parameter1 = 0.95
    parameter2 = 1.05

    uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)
end

cd(@__DIR__)

begin
    @load "fvm-fine/fvm1.jld2" x
    xc = deepcopy(x)

    fvm = []
    for i = 1:9
        filename = "fvm-fine/fvm" * string(i) * ".jld2"
        @load filename x sol
        push!(fvm, sol)
    end

    solc = zeros(size(fvm[1])..., 2)
    for i in axes(solc, 1), j in axes(solc, 2), k in axes(solc, 3)
        uRan = [fvm[idx][i, j, k] for idx = 1:9]
        uChaos = ran_chaos(uRan, uq)

        solc[i, j, k, 1] = mean(uChaos, uq.op)
        solc[i, j, k, 2] = std(uChaos, uq.op)
    end
end

begin
    @load "fvm/fvm1.jld2" x
    xr = deepcopy(x)

    fvm = []
    for i = 1:9
        filename = "fvm/fvm" * string(i) * ".jld2"
        @load filename x sol
        push!(fvm, sol)
    end

    solr = zeros(size(fvm[1])..., 2)
    for i in axes(solr, 1), j in axes(solr, 2), k in axes(solr, 3)
        uRan = [fvm[idx][i, j, k] for idx = 1:9]
        uChaos = ran_chaos(uRan, uq)

        solr[i, j, k, 1] = mean(uChaos, uq.op)
        solr[i, j, k, 2] = std(uChaos, uq.op)
    end
end

@load "galerkin/sol.jld2" x u#sol
begin
    sol = zeros(size(x)..., 4, 2)
    for i = 1:ps.nx, j = 1:ps.ny
        idx0 = (i - 1) * (ps.deg+1)
        idy0 = (j - 1) * (ps.deg+1)

        for k = 1:ps.deg+1, l = 1:ps.deg+1
            idx = idx0 + k
            idy = idy0 + l

            uChaos = uq_conserve_prim(u[i, j, k, l, :, :], ks.gas.γ, uq)
            uChaos[4, :] .= lambda_tchaos(Array(uChaos[4, :]), 1.0, uq)

            for s = 1:4
                sol[idx, idy, s, 1] = mean(uChaos[s, :], uq.op)
                sol[idx, idy, s, 2] = std(uChaos[s, :], uq.op)
            end
        end
    end
end

# density
plot(
    x[:, 1],
    0.5 .* (sol[:, end÷2, 1, 1] .+ sol[:, end÷2+1, 1, 1]),
    lw = 1.5,
    label = "Current",
    xlabel = "x",
    ylabel = "density",
)
scatter!(
    xc[1:2:end, 1],
    0.5 .* (solc[1:2:end, end÷2, 1, 1] .+ solc[1:2:end, end÷2+1, 1, 1]),
    label = "Collocation",
    alpha = 0.8,
)
plot!(
    xr,
    0.5 .* (solr[:, end÷2, 1, 1] .+ solr[:, end÷2+1, 1, 1]),
    lw = 1.5,
    label = "FVM",
    line = :dash,
    color = :gray27,
)
savefig("sv_t1_n_mean.pdf")

plot(
    x[:, 1],
    0.5 .* (sol[:, end÷2, 1, 2] .+ sol[:, end÷2+1, 1, 2]),
    lw = 1.5,
    label = "Current",
    xlabel = "x",
    ylabel = "density",
)
scatter!(
    xc[1:2:end, 1],
    0.5 .* (solc[1:2:end, end÷2, 1, 2] .+ solc[1:2:end, end÷2+1, 1, 2]),
    label = "Collocation",
    alpha = 0.8,
)
plot!(
    xr,
    0.5 .* (solr[:, end÷2, 1, 2] .+ solr[:, end÷2+1, 1, 2]),
    lw = 1.5,
    label = "FVM",
    line = :dash,
    color = :gray27,
)
savefig("sv_t1_n_std.pdf")

# temperature
plot(
    x[:, 1],
    0.5 .* (sol[:, end÷2, 4, 1] .+ sol[:, end÷2+1, 4, 1]),
    lw = 1.5,
    label = "Current",
    xlabel = "x",
    ylabel = "temperature",
)
scatter!(
    xc[1:2:end, 1],
    0.5 .* (solc[1:2:end, end÷2, 4, 1] .+ solc[1:2:end, end÷2+1, 4, 1]),
    label = "Collocation",
    alpha = 0.8,
)
plot!(
    xr,
    0.5 .* (solr[:, end÷2, 4, 1] .+ solr[:, end÷2+1, 4, 1]),
    lw = 1.5,
    label = "FVM",
    line = :dash,
    color = :gray27,
)
savefig("sv_t1_t_mean.pdf")

plot(
    x[:, 1],
    0.5 .* (sol[:, end÷2, 4, 2] .+ sol[:, end÷2+1, 4, 2]),
    lw = 1.5,
    label = "Current",
    xlabel = "x",
    ylabel = "temperature",
)
scatter!(
    xc[1:2:end, 1],
    0.5 .* (solc[1:2:end, end÷2, 4, 2] .+ solc[1:2:end, end÷2+1, 4, 2]),
    label = "Collocation",
    alpha = 0.8,
)
plot!(
    xr,
    0.5 .* (solr[:, end÷2, 4, 2] .+ solr[:, end÷2+1, 4, 2]),
    lw = 1.5,
    label = "FVM",
    line = :dash,
    color = :gray27,
)
savefig("sv_t1_t_std.pdf")

# ---
# contour
# ---
@load "iter_0.3.jld2" u#sol
begin
    sol = zeros(size(x)..., 4, 2)
    for i = 1:ps.nx, j = 1:ps.ny
        idx0 = (i - 1) * (ps.deg+1)
        idy0 = (j - 1) * (ps.deg+1)

        for k = 1:ps.deg+1, l = 1:ps.deg+1
            idx = idx0 + k
            idy = idy0 + l

            uChaos = uq_conserve_prim(u[i, j, k, l, :, :], ks.gas.γ, uq)
            uChaos[4, :] .= lambda_tchaos(Array(uChaos[4, :]), 1.0, uq)

            for s = 1:4
                sol[idx, idy, s, 1] = mean(uChaos[s, :], uq.op)
                sol[idx, idy, s, 2] = std(uChaos[s, :], uq.op)
            end
        end
    end
end

contourf(
    x[:, 1],
    y[1, :],
    sol[:, :, 1, 1]',
    ratio=1,
    ylims=(0,1),
    xlabel="x",
    ylabel="y",
)

savefig("sv_t03_mean.pdf")

contourf(
    x[:, 1],
    y[1, :],
    sol[:, :, 1, 2]',
    ratio=1,
    ylims=(0,1),
    xlabel="x",
    ylabel="y",
)

savefig("sv_t03_std.pdf")
