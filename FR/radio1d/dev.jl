using OrdinaryDiffEq, LinearAlgebra, Plots
using KitBase, FluxReconstruction
using ProgressMeter: @showprogress

set = Setup(
    matter = "radiation",
    case = "inflow",
    space = "1d1f1v",
    boundary = "maxwell",
    cfl = 0.2,
    maxTime = 0.1,
)

deg = 2
ps = FRPSpace1D(0, 1, 50, deg)
vs = VSpace1D(-1, 1, 28)

δ = heaviside.(vs.u)

