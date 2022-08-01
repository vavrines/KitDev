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

function st!(KS, uq, faceL, cell, faceR, p, coll = :bgk)
    dt, dx, RES, AVG = p

    w_old = deepcopy(cell.w)
    prim_old = deepcopy(cell.prim)

    @. cell.w += (faceL.fw - faceR.fw) / dx

    wRan = chaos_ran(cell.w, 2, uq)

    primRan = zero(wRan)
    for j in axes(wRan, 2)
        primRan[:, j] .= quantum_conserve_prim(wRan[:, j], KS.gas.γ)
    end

    #cell.w .= ran_chaos(wRan, 2, uq)
    cell.prim .= ran_chaos(primRan, 2, uq)

    #@. cell.f += (faceL.ff - faceR.ff) / cell.dx
    fRan =
        chaos_ran(cell.f, 2, uq) .+
        (chaos_ran(faceL.ff, 2, uq) .- chaos_ran(faceR.ff, 2, uq)) ./ dx

    tau = [KS.gas.Kn / wRan[1, j] for j in axes(wRan, 2)]

    gRan = zeros(KS.vSpace.nu, uq.op.quad.Nquad)
    for j in axes(gRan, 2)
        gRan[:, j] .= fd_equilibrium(KS.vs.u, primRan[:, j], ks.gas.γ)
    end

    for j in axes(fRan, 2)
        @. fRan[:, j] = (fRan[:, j] + dt / tau[j] * gRan[:, j]) / (1.0 + dt / tau[j])
    end

    cell.f .= ran_chaos(fRan, 2, uq)

    @. RES += (w_old[:, 1] - cell.w[:, 1])^2
    @. AVG += abs(cell.w[:, 1])

    return nothing
end
