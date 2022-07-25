using Plots, JLD2

cd(@__DIR__)
begin
    u0 = []
    @load "exp.jld2" x sol
    push!(u0, sol)
    @load "l2.jld2" x sol
    push!(u0, sol)
    @load "lasso.jld2" x sol
    push!(u0, sol)
    @load "nofilter.jld2" x sol
    push!(u0, sol)

    u = []
    @load "l2_apt.jld2" x sol
    push!(u, sol)
end

plot(
    x,
    u0[end][:, 2],
    label = "No filter",
    lw = 1.5,
    color = :gray27,
    xlabel = "x",
    ylabel = "standard deviation",
)
plot!(x, u0[1][:, 2], label = "Exp", lw = 1.5, color = 1)
plot!(x, u0[3][:, 2], label = "Lasso", lw = 1.5, color = 2)
plot!(x, u0[2][:, 2], label = "L²", lw = 1.5, color = 3)
plot!(x, u[1][:, 2], label = "Adaptive L²", lw = 1.5, color = 4)
savefig("burgers_std.pdf")

plot(
    x,
    u0[end][:, 1],
    label = "No filter",
    lw = 1.5,
    color = :gray27,
    xlabel = "x",
    ylabel = "expected value",
)
plot!(x, u0[1][:, 1], label = "Exp", lw = 1.5, color = 1)
plot!(x, u0[3][:, 1], label = "Lasso", lw = 1.5, color = 2)
plot!(x, u0[2][:, 1], label = "L²", lw = 1.5, color = 3)
plot!(x, u[1][:, 1], label = "Adaptive L²", lw = 1.5, color = 4)
savefig("burgers_mean.pdf")
