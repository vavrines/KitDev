using OrdinaryDiffEq, CairoMakie, Langevin
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

tend, Δt = 10.0, 0.01
tSpace = Vector(0:Δt:tend)
tnum = Int(tend / Δt + 1)

unum = 200
vSpace = VSpace1D(-6.0, 6.0, unum, "rectangle")
γ = 3
f0 = vSpace.u .^ 2 .* exp.(-vSpace.u .^ 2)
w0 = moments_conserve(f0, vSpace.u, vSpace.weights)
prim0 = conserve_prim(w0, γ)
g0 = maxwellian(vSpace.u, prim0)

μ, σ = 1.0, 0.2 # gaussian

#--- theoretical solution ---#
fMeanExa = zeros(tnum, unum)
fStdExa = zeros(tnum, unum)
for i = 1:tnum
    for j = 1:unum
        fMeanExa[i, j] =
            f0[j] * exp(-tSpace[i] + 0.5 * (tSpace[i] * σ)^2) +
            g0[j] * (1.0 - exp(-tSpace[i] + 0.5 * (tSpace[i] * σ)^2))
        fStdExa[i, j] = sqrt(
            (f0[j] - g0[j])^2 *
            (exp(tSpace[i]^2 * σ^2) - 1.0) *
            exp(-tSpace[i] * 2 + tSpace[i]^2 * σ^2),
        )
    end
end

contour(vSpace.u, tSpace, fMeanExa)
contour(vSpace.u, tSpace, fStdExa, fill = true)








#--- numerical solution ---#
L, Nrec = 5, 40
op = GaussOrthoPoly(L, Nrec = Nrec)

a = [convert2affinePCE(μ, σ, op); zeros(L - 1)] # collision frequency

finit = zeros(L + 1, unum)
finit[1, :] .= f0

t1 = Tensor(1, op); # < \phi_i >
t2 = Tensor(2, op); # < \phi_i, \phi_j >
t3 = Tensor(3, op); # < \phi_i \phi_j, \phi_k >

function ODEGalerkinTen(du, u, p, t)
    for m = 0:L
        for i = 1:unum
            du[m+1, i] = (
                g0[i] * p[m+1] - sum(
                    p[j+1] * u[k+1, i] * t3.get([j, k, m]) / t2.get([m, m]) for j = 0:L
                    for k = 0:L
                )
            )
        end
    end
end

probGalerkinTen = ODEProblem(ODEGalerkinTen, finit, (0, tend), a)
solGalerkinTen =
    solve(probGalerkinTen, Tsit5(), abstol = 1e-10, reltol = 1e-10, saveat = 0:Δt:tend)

#--- analysis ---#
solTen = zeros(tnum, L + 1, unum)
for i = 1:tnum
    solTen[i, :, :] = solGalerkinTen.u[i]
end

fMeanNum = zeros(tnum, unum)
fStdNum = zeros(tnum, unum)
for i = 1:tnum
    for j = 1:unum
        fMeanNum[i, j] = mean(solTen[i, :, j], op)
        fStdNum[i, j] = std(solTen[i, :, j], op)
    end
end

erL1Mean = sum(abs.(fMeanExa .- fMeanNum)) .* Δt .* (vSpace.u[unum] - vSpace.u[unum-1])
erL2Mean =
    sqrt(sum(((fMeanExa .- fMeanNum) .* Δt .* (vSpace.u[unum] - vSpace.u[unum-1])) .^ 2))

contour(vSpace.u, tSpace, fMeanNum, fill = true)
contour(vSpace.u, tSpace, fStdNum, fill = true)
