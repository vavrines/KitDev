using Plots, JLD2, Langevin

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

fvm = []
for i = 1:9
    filename = "fvm/fvm" * string(i) * ".jld2"
    @load filename x sol
    push!(fvm, sol)
end
x_ref = deepcopy(x)

sol_ref = zeros(size(fvm[1])..., 2)
for  i in axes(sol_ref, 1), j in axes(sol_ref, 2), k in axes(sol_ref, 3)
    uRan = [fvm[idx][i, j, k] for idx = 1:9]
    uChaos = ran_chaos(uRan, uq)

    sol_ref[i, j, k, 1] = mean(uChaos, uq.op)
    sol_ref[i, j, k, 2] = std(uChaos, uq.op)
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

@load "collocation/sol.jld2" x u#sol
begin
    sol1 = zeros(size(x)..., 4, 2)
    for i = 1:ps.nx, j = 1:ps.ny
        idx0 = (i - 1) * (ps.deg+1)
        idy0 = (j - 1) * (ps.deg+1)

        for k = 1:ps.deg+1, l = 1:ps.deg+1
            idx = idx0 + k
            idy = idy0 + l

            uRan = uq_conserve_prim(u[i, j, k, l, :, :], ks.gas.γ, uq)
            uChaos = zeros(4, uq.nm+1)
            for ii = 1:4
                uChaos[ii, :] .= ran_chaos(uRan[ii, :], uq)
            end
            uChaos[4, :] .= lambda_tchaos(Array(uChaos[4, :]), 1.0, uq)

            for s = 1:4
                sol1[idx, idy, s, 1] = mean(uChaos[s, :], uq.op)
                sol1[idx, idy, s, 2] = std(uChaos[s, :], uq.op)
            end
        end
    end
end

plot(
    x[:, 1],
    0.5 .* (sol[:, end÷2, 1, 1] .+ sol[:, end÷2+1, 1, 1]),
    lw = 1.5,
    label = "Current",
    xlabel = "x",
    ylabel = "density",
)
scatter!(
    x[1:4:end, 1],
    0.5 .* (sol1[1:4:end, end÷2, 1, 1] .+ sol1[1:4:end, end÷2+1, 1, 1]),
    lw = 2,
    label = "Collocation",
    alpha = 0.75,
)
plot!(
    x_ref,
    0.5 .* (sol_ref[:, end÷2, 1, 1] .+ sol_ref[:, end÷2+1, 1, 1]),
    lw = 2,
    label = "FVM",
    line = :dash,
    color = :gray27,
)
savefig("sv_t1_n_mean.pdf")

plot(
    x[:, 1],
    0.5 .* (sol[:, end÷2, 1, 2] .+ sol[:, end÷2+1, 1, 2]),
    lw = 2,
    label = "FR",
    xlabel = "x",
    ylabel = "density",
)
plot!(
    x_ref,
    0.5 .* (sol_ref[:, end÷2, 1, 2] .+ sol_ref[:, end÷2+1, 1, 2]),
    lw = 2,
    label = "FVM",
    line = :dash,
    color = :gray27,
)
savefig("sv_t1_n_std.pdf")

plot(
    x[:, 1],
    0.5 .* (sol[:, end÷2, 4, 1] .+ sol[:, end÷2+1, 4, 1]),
    lw = 2,
    label = "FR",
    xlabel = "x",
    ylabel = "temperature",
)
plot!(
    x_ref,
    0.5 .* (sol_ref[:, end÷2, 4, 1] .+ sol_ref[:, end÷2+1, 4, 1]),
    lw = 2,
    label = "FVM",
    line = :dash,
    color = :gray27,
)
savefig("sv_t1_t_mean.pdf")

plot(
    x[:, 1],
    0.5 .* (sol[:, end÷2, 4, 2] .+ sol[:, end÷2+1, 4, 2]),
    lw = 2,
    label = "FR",
    xlabel = "x",
    ylabel = "temperature",
)
plot!(
    x_ref,
    0.5 .* (sol_ref[:, end÷2, 4, 2] .+ sol_ref[:, end÷2+1, 4, 2]),
    lw = 2,
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
