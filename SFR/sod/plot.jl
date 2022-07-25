using JLD2, Plots

cd(@__DIR__)
begin
    @load "results/collocation.jld2" x sol1 sol2
    x0 = deepcopy(x)
    sol_collo = [sol1, sol2]

    @load "results/l2.jld2" x sol1 sol2
    sol_l2 = [sol1, sol2]

    @load "results/l2_apt.jld2" x sol1 sol2
    sol_apt = [sol1, sol2]

    @load "results/lasso.jld2" x sol1 sol2
    sol_lasso = [sol1, sol2]
end

#--- case 1 ---#
plot(x, sol_lasso[1][:, 1, 1], lw = 2, label = "Lasso", xlabel = "x", ylabel = "density")
plot!(x, sol_l2[1][:, 1, 1], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[1][:, 1, 1],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod1_density_mean.pdf")

plot(
    x,
    sol_lasso[1][:, 2, 1],
    lw = 2,
    label = "Lasso",
    xlabel = "x",
    ylabel = "velocity",
    legend = :topleft,
)
plot!(x, sol_l2[1][:, 2, 1], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[1][:, 2, 1],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod1_velocity_mean.pdf")

plot(
    x,
    sol_lasso[1][:, 3, 1],
    lw = 2,
    label = "Lasso",
    xlabel = "x",
    ylabel = "temperature",
    legend = :topleft,
)
plot!(x, sol_l2[1][:, 3, 1], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[1][:, 3, 1],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod1_temperature_mean.pdf")

plot(x, sol_lasso[1][:, 1, 2], lw = 2, label = "Lasso", xlabel = "x", ylabel = "density")
plot!(x, sol_l2[1][:, 1, 2], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[1][:, 1, 2],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod1_density_std.pdf")

plot(
    x,
    sol_lasso[1][:, 2, 2],
    lw = 2,
    label = "Lasso",
    legend = :topleft,
    xlabel = "x",
    ylabel = "velocity",
)
plot!(x, sol_l2[1][:, 2, 2], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[1][:, 2, 2],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod1_velocity_std.pdf")

plot(
    x,
    sol_lasso[1][:, 3, 2],
    lw = 2,
    label = "Lasso",
    xlabel = "x",
    ylabel = "temperature",
    legend = :topleft,
)
plot!(x, sol_l2[1][:, 3, 2], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[1][:, 3, 2],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod1_temperature_std.pdf")

#--- case 2 ---#
plot(x, sol_lasso[2][:, 1, 1], lw = 2, label = "Lasso", xlabel = "x", ylabel = "density")
plot!(x, sol_l2[2][:, 1, 1], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[2][:, 1, 1],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod2_density_mean.pdf")

plot(x, sol_lasso[2][:, 2, 1], lw = 2, label = "Lasso", xlabel = "x", ylabel = "velocity")
plot!(x, sol_l2[2][:, 2, 1], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[2][:, 2, 1],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod2_velocity_mean.pdf")

plot(
    x,
    sol_lasso[2][:, 3, 1],
    lw = 2,
    label = "Lasso",
    xlabel = "x",
    ylabel = "temperature",
    legend = :topleft,
)
plot!(x, sol_l2[2][:, 3, 1], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[2][:, 3, 1],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod2_temperature_mean.pdf")

plot(x, sol_lasso[2][:, 1, 2], lw = 2, label = "Lasso", xlabel = "x", ylabel = "density")
plot!(x, sol_l2[2][:, 1, 2], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[2][:, 1, 2],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod2_density_std.pdf")

plot(
    x,
    sol_lasso[2][:, 2, 2],
    lw = 2,
    legend = :topleft,
    label = "Lasso",
    xlabel = "x",
    ylabel = "density",
)
plot!(x, sol_l2[2][:, 2, 2], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[2][:, 2, 2],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod2_velocity_std.pdf")

plot(x, sol_lasso[2][:, 3, 2], lw = 2, label = "Lasso", xlabel = "x", ylabel = "density")
plot!(x, sol_l2[2][:, 3, 2], lw = 2, label = "Adaptive L²", line = :dash)
plot!(
    x0,
    sol_collo[2][:, 3, 2],
    label = "Reference",
    line = :dashdot,
    lw = 2,
    color = :gray27,
)
savefig("sod2_temperature_std.pdf")
