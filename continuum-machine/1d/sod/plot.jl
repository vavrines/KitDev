using Kinetic, Plots, JLD2

cd(@__DIR__)

ctrs = []
begin
    @load "nn.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kngll.jld2" ks ctr
    push!(ctrs, ctr)
    @load "pure_kinetic.jld2" ks ctr
    push!(ctrs, ctr)
    @load "pure_ns.jld2" ks ctr
    push!(ctrs, ctr)
end

sols = []
begin
    for idx in eachindex(ctrs)
        sol = zeros(ks.ps.nx, 3)
        for i in axes(sol, 1)
            sol[i, :] .= conserve_prim(ctrs[idx][i].prim, ks.gas.γ)
            sol[i, end] = 1 / sol[i, end]
        end
        push!(sols, sol)
    end
end








idx = ks.ps.nx÷5*3+1:ks.ps.nx÷5*4

plot(ks.ps.x[idx], sols[1][idx, 1], lw=1.5, label="NN", xlabel="x", ylabel="ρ")
plot!(ks.ps.x[idx], sols[2][idx, 1], lw=1.5, label="KnGLL")
plot!(ks.ps.x[idx], sols[3][idx, 1], lw=1.5, label="kinetic")
plot!(ks.ps.x[idx], sols[4][idx, 1], lw=1.5, label="NS")

plot(ks.ps.x[idx], sols[1][idx, 3], lw=1.5, label="NN", xlabel="x", ylabel="T")
plot!(ks.ps.x[idx], sols[2][idx, 3], lw=1.5, label="KnGLL")
plot!(ks.ps.x[idx], sols[3][idx, 3], lw=1.5, label="kinetic")
plot!(ks.ps.x[idx], sols[4][idx, 3], lw=1.5, label="NS")
