include("polylog.jl")


"""
    be_integral(ν, z)

Bose-Einstein integral
"""
be_integral(ν, z) = polylog(ν+1, z) |> real


"""
    fd_integral(ν, z)

Fermi-Dirac integral
"""
fd_integral(ν, z) = -polylog(ν+1, -z) |> real


function be_equilibrium(u, prim, β)
    A, U, λ = prim
    return @. β / sqrt(π) * (A^(-1) * exp(λ * (u - U)^2) - 1)^(-1)
end

function fd_equilibrium(u, prim, β)
    A, U, λ = prim
    return @. β / sqrt(π) * (A^(-1) * exp(λ * (u - U)^2) + 1)^(-1)
end

