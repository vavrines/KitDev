using KitBase, Plots
using KitBase.JLD2

cd(@__DIR__)

begin
    set = Setup(space = "1d2f1v", nSpecies = 2, boundary = "maxwell", maxTime = 10)
    ps = PSpace1D(0.0, 1.0, 100)
    vs = MVSpace1D(-5, 5, -8, 8, 72)
    gas = Mixture(Kn = 1e-0, K = 2.0)

    function ib_condition(set, ps, vs, gas, T = (1.0, 0.8), P = 1.0)
        primL = zeros(3, 2)
        primL[1, 1] = gas.mi
        primL[1, 2] = gas.me
        primL[3, 1] = gas.mi / T[1]
        primL[3, 2] = gas.me / T[1]

        primR = zeros(3, 2)
        primR[1, 1] = gas.mi
        primR[1, 2] = gas.me
        primR[3, 1] = gas.mi / T[2]
        primR[3, 2] = gas.me / T[2]

        p = (x0 = ps.x0, x1 = ps.x1, primL = primL, primR = primR, γ = gas.γ, u = vs.u)

        fw = function (x, p)
            prim = zeros(3, 2)
            prim[1, 1] = gas.mi
            prim[1, 2] = gas.me
            prim[3, 1] = gas.mi
            prim[3, 2] = gas.me

            return mixture_prim_conserve(prim, p.γ)
        end

        bc = function (x, p)
            if x <= (p.x0 + p.x1) / 2
                return p.primL
            else
                return p.primR
            end
        end

        ff = function (x, p)
            w = fw(x, p)
            prim = mixture_conserve_prim(w, p.γ)
            h = mixture_maxwellian(p.u, prim)
            b = similar(h)
            for j = 1:2
                b[:, j] .= h[:, j] .* gas.K ./ (2.0 .* prim[end, j])
            end

            return h, b
        end

        return fw, ff, bc, p
    end

    fw, ff, bc, p = ib_condition(set, ps, vs, gas)
    ib = IB2F(fw, ff, bc, p)

    ks = KB.SolverSet(set, ps, vs, gas, ib)
end

@load "ctr_kn0.jld2" ctr
@load "ctr_kn1.jld2" ctr

sol = zeros(ks.ps.nx, 3, 2)
for i in axes(sol, 1)
    sol[i, :, :] .= ctr[i].prim
end

plot(ks.ps.x, sol[:, 1, 1] ./ ks.gas.mi)
plot!(ks.ps.x, sol[:, 1, 2] ./ ks.gas.me)

plot(ks.ps.x, ks.gas.mi ./ sol[:, 3, 1])
plot!(ks.ps.x, ks.gas.me ./ sol[:, 3, 2])
