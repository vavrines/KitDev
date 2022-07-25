using NonlinearSolve, SciMLNLSolve

#include("polylog.jl")

"""
    be_integral(ν, z)

Bose-Einstein integral
"""
be_integral(ν, z) = polylog(ν + 1, z) |> real


"""
    fd_integral(ν, z)

Fermi-Dirac integral
"""
fd_integral(ν, z) = -polylog(ν + 1, -z) |> real


function be_equilibrium(u, prim, β)
    A, U, λ = prim
    return @. β / sqrt(π) * (A^(-1) * exp(λ * (u - U)^2) - 1)^(-1)
end

function fd_equilibrium(u, prim, β)
    A, U, λ = prim
    return @. β / sqrt(π) * (A^(-1) * exp(λ * (u - U)^2) + 1)^(-1)
end


function Aeq(u, p)
    ρ, ρe, β = p
    return 4 * β^2 * fd_integral(-0.5, u)^3 * ρe - ρ^3 * fd_integral(0.5, u)
end
const Aprob0 = NonlinearProblem{false}(Aeq, 0.3, (0.556, 0.19, 2))
Aprob(w, β) = remake(Aprob0, u0 = w[1], p = (w[1], w[3] - w[2]^2 / w[1] / 2, β))


function quantum_conserve_prim(w, β)
    prim = zero(w)

    prob = Aprob(w, β)
    sol = solve(prob, NLSolveJL())
    prim[1] = sol.u[1]

    prim[2] = w[2] / w[1]
    prim[3] = β^2 * fd_integral(-0.5, prim[1])^2 / w[1]^2

    return prim
end

function quantum_prim_conserve(prim, β)
    w = zero(prim)

    w[1] = fd_integral(-0.5, prim[1]) * β / sqrt(prim[3])
    w[2] = w[1] * prim[2]
    w[3] = fd_integral(0.5, prim[1]) * β / (prim[3])^(3 / 2) / 4 + 0.5 * w[1] * prim[2]^2

    return w
end
