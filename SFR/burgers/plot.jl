using Plots, JLD2

cd(@__DIR__)
begin
    u0 = []
    @load "l2.jld2" x sol
    push!(u0, sol)
    @load "l2opt.jld2" x sol
    push!(u0, sol)
    @load "lasso.jld2" x sol
    push!(u0, sol)
    @load "nofilter.jld2" x sol
    push!(u0, sol)

    u = []
    @load "l2_apt.jld2" x sol
    push!(u, sol)
    @load "l2opt_apt.jld2" x sol
    push!(u, sol)
    @load "lasso_apt.jld2" x sol
    push!(u, sol)
end

str = 1

plot(x, u[3][:, 1], label="Lasso", lw=2, xlabel="x", ylabel="expected value")
plot!(x, u[2][:, 1], label="Adaptive L²", lw=2, line=:dash)
plot!(x, u0[2][:, 1], label="Optimized L²", lw=2, line=:dashdot)
plot!(x, u0[1][:, 1], label="Standard L²", lw=2, line=:dashdotdot)
#scatter!(x[1:str:end], u0[4][1:str:end, 1], label="No filter", markeralpha=0.6, color=4)
plot!(x, u0[4][:, 1], label="No filter", lw=2, line=:dot, color=:gray27)
savefig("burgers_mean.pdf")

plot(x, u[3][:, 2], label="Lasso", lw=2, xlabel="x", ylabel="standard deviation")
plot!(x, u[2][:, 2], label="Adaptive L²", lw=2, line=:dash)
plot!(x, u0[2][:, 2], label="Optimized L²", lw=2, line=:dashdot)
plot!(x, u0[1][:, 2], label="Standard L²", lw=2, line=:dashdotdot)
#scatter!(x[1:str:end], u0[4][1:str:end, 2], label="No filter", markeralpha=0.6, color=4)
plot!(x, u0[4][:, 2], label="No filter", lw=2, line=:dot, color=:gray27)
savefig("burgers_std.pdf")
