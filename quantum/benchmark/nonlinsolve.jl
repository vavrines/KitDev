using KitBase.SpecialFunctions
using BenchmarkTools

cd(@__DIR__)
include("../tools.jl")

w = [1.0, 0.25, 1.25]
β = 2.0

# conserve_prim(w, β)

prob = Aprob(w, β)

u1 = solve(prob, NLSolveJL(method=:trust_region)) # default
u2 = solve(prob, NLSolveJL(method=:anderson))

@btime solve(prob, NLSolveJL(method=:trust_region))
@btime solve(prob, NLSolveJL(method=:anderson))
