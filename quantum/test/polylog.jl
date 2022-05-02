# ============================================================
# Test of polylogarithm function
# ============================================================

using PyCall

cd(@__DIR__)
include("../polylog.jl")

py"""
import mpmath as mp
import sympy as sp

mp.dps = 15; mp.pretty = True

def polylog(s, z):
    return mp.polylog(s, z)
"""

begin
    for s in [1, 2, 3, 4, 0.5, 1.5, 2.5, 3.5]
        for z in [0.5, -0.5]
            x = polylog(s, z)
            x0 = py"polylog"(s, z)
            abs(x - x0) > 1e-10 && println("s=", s, ", z=", z)
        end
    end
end

using BenchmarkTools

@btime polylog(1.5, 0.9)
@btime py"polylog"(1.5, 0.9)

@btime polylog(1.5, 0.9)
@btime be_integral(0.5, 0.9)
