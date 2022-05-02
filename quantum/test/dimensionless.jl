using NonlinearSolve, SciMLNLSolve, CairoMakie
import KitBase as KB

cd(@__DIR__)
include("../tools.jl")

β = 2
vs = KB.VSpace1D(-5, 5, 100)
prim = [0.3, 0.3, 0.8]

f = fd_equilibrium(vs.u, prim, β)

begin
    fig = lines(vs.u, f; label = "f")
    axislegend()
    fig
end

w1 = KB.moments_conserve(f, vs.u, vs.weights)
w2 = prim_conserve(prim, β)
prim1 = conserve_prim(w2, β)

prob = Aprob(w1, β)
sol = solve(prob, NLSolveJL())
