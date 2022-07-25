using Kinetic, Plots, JLD2, LinearAlgebra
using Flux: onecold

cd(@__DIR__)

# Kn = 1e-4
ctrs = []
begin
    @load "kn=1e-4/nn.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-4/kngll.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-4/pure_kinetic.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-4/pure_ns.jld2" ks ctr
    push!(ctrs, ctr)
end

# Kn = 1e-3
ctrs = []
begin
    @load "kn=1e-3/nn.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-3/kngll.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-3/pure_kinetic.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-3/pure_ns.jld2" ks ctr
    push!(ctrs, ctr)
end

# Kn = 1e-2
ctrs = []
begin
    @load "kn=1e-2/nn.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-2/kngll.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-2/pure_kinetic.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-3/pure_ns.jld2" ks ctr
    push!(ctrs, ctr)
    @load "kn=1e-2/pure_kinetic.jld2" ks ctr
end

sols = []
begin
    for idx in eachindex(ctrs)
        sol = zeros(ks.ps.nx, 3)
        for i in axes(sol, 1)
            sol[i, :] .= ctrs[idx][i].prim
            sol[i, end] = 1 / sol[i, end]
        end
        push!(sols, sol)
    end
end

function regime_data(ks, w, prim, sw, f)
    Mu, Mxi, _, _1 = gauss_moments(prim, ks.gas.K)
    a = pdf_slope(prim, sw, ks.gas.K)
    sw = -prim[1] .* moments_conserve_slope(a, Mu, Mxi, 1)
    A = pdf_slope(prim, sw, ks.gas.K)
    tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
    fr = chapman_enskog(ks.vs.u, prim, a, A, tau)
    L = norm((f .- fr) ./ prim[1])

    x = [w; sw; tau]
    y = ifelse(L <= 0.005, [1.0, 0.0], [0.0, 1.0])
    return x, y
end

begin
    ctr = ctrs[3]

    rg_ce = zeros(Int, ks.ps.nx)
    for i = 1:ks.ps.nx
        sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
        x, y = regime_data(ks, ctr[i].w, ctr[i].prim, sw, ctr[i].h)
        rg_ce[i] = onecold(y)
    end

    rg_kn = zeros(Int, ks.ps.nx)
    KnGLL = zeros(ks.ps.nx)
    for i = 1:ks.ps.nx
        sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
        L = abs(ctr[i].w[1] / sw[1])
        ℓ =
            (1 / ctr[i].prim[end])^ks.gas.ω / ctr[i].prim[1] *
            sqrt(ctr[i].prim[end]) *
            ks.gas.Kn

        KnGLL[i] = ℓ / L
        rg_kn[i] = ifelse(KnGLL[i] >= 0.05, 2, 1)
    end
end

scatter(
    ks.ps.x[1:ks.ps.nx],
    rg_ce,
    alpha = 0.7,
    label = "NN",
    xlabel = "x",
    ylabel = "regime",
)
scatter!(ks.ps.x[1:ks.ps.nx], rg_kn, alpha = 0.7, label = "KnGLL")
plot!(
    ks.ps.x[1:ks.ps.nx],
    rg_ce,
    lw = 1.5,
    line = :dot,
    color = :gray27,
    label = "True",
    xlabel = "x",
    ylabel = "regime",
)

#savefig("kn4_regime.pdf")
#savefig("kn3_regime.pdf")
#savefig("kn2_regime.pdf")

function curve(idx, s, lgd = :topright; ns = true)
    fig = plot(ks.ps.x[idx], sols[1][idx, s], lw = 1.5, label = "NN", legend = lgd)
    plot!(fig, ks.ps.x[idx], sols[2][idx, s], lw = 1.5, line = :dash, label = "KnGLL")
    plot!(fig, ks.ps.x[idx], sols[3][idx, s], lw = 1.5, line = :dashdot, label = "Kinetic")
    ns && plot!(
        fig,
        ks.ps.x[idx],
        sols[4][idx, s],
        lw = 2,
        line = :dot,
        color = :gray27,
        label = "NS",
    )

    xlabel!(fig, "x")
    if s == 1
        ylabel!(fig, "ρ")
    elseif s == 2
        ylabel!(fig, "U")
    elseif s == 3
        ylabel!(fig, "T")
    end

    return fig
end

idx = 1:ks.ps.nx
curve(idx, 1)
#savefig("kn4_n.pdf")
#savefig("kn3_n.pdf")
#curve(idx, 1; ns=false)
#savefig("kn2_n.pdf")

curve(idx, 3)
#savefig("kn4_t.pdf")
#savefig("kn3_t.pdf")
#curve(idx, 3; ns=false)
#savefig("kn2_t.pdf")

idx = ks.ps.nx÷5*3+1:ks.ps.nx÷5*4
curve(idx, 1)
#savefig("kn4_n_zoom.pdf")
#savefig("kn3_n_zoom.pdf")
#curve(idx, 1; ns=false)
#savefig("kn2_n_zoom.pdf")

curve(idx, 3, :topleft)
#savefig("kn4_t_zoom.pdf")
#savefig("kn3_t_zoom.pdf")
#curve(idx, 3, :topleft; ns=false)
#savefig("kn2_t_zoom.pdf")
