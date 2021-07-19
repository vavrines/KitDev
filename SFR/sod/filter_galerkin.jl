using KitBase, FluxReconstruction, OrdinaryDiffEq, Langevin, LinearAlgebra, Plots
using ProgressMeter: @showprogress

cd(@__DIR__)
include("rhs.jl")

function FR.modal_filter!(u::AbstractMatrix{T}, args...; filter::Symbol) where {T<:AbstractFloat}
    filtstr = "filter_" * string(filter) * "!"
    filtfunc = Symbol(filtstr) |> eval
    filtfunc(u, args...)

    return nothing
end

function FR.filter_l2!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    p0 = axes(u, 1) |> first
    p1 = axes(u, 1) |> last
    q0 = axes(u, 2) |> first
    q1 = axes(u, 2) |> last
    @assert p0 >= 0
    @assert q0 >= 0

    λx, λξ = args[1:2]
    for j in axes(u, 2), i in axes(u, 1)
        u[i, j] /= (1.0 + λx * (i - p0 + 1)^2 * (i - p0)^2 + λξ * (j - q0 + 1)^2 * (j - q0)^2)
    end

    return nothing
end

function FR.filter_exp!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    nx, nz = size(u)
    λx, λξ = args[1:2]

    if length(args) >= 3
        Ncx = args[3]
    else
        Ncx = 0
    end
    if length(args) >= 4
        Ncξ = args[4]
    else
        Ncξ = 0
    end

    σ1 = FR.filter_exp1d(nx-1, λx, Ncx)
    σ2 = FR.filter_exp1d(nz-1, λξ, Ncξ)

    for j in axes(u, 2), i in axes(u, 1)
        u[i, j] *= σ1[i] * σ2[j]
    end

    return nothing
end

function filter_houli!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    nx, nz = size(u)
    λx, λξ = args[1:2]

    if length(args) >= 3
        Ncx = args[3]
    else
        Ncx = 0
    end
    if length(args) >= 4
        Ncξ = args[4]
    else
        Ncξ = 0
    end

    σ1 = FR.filter_exp1d(nx-1, λx, Ncx)
    σ2 = FR.filter_exp1d(nz-1, λξ, Ncξ)

    for i in eachindex(σ1)
        if i/length(σ1) <= 2/3
            σ1[i] = 1.0
        end
    end
    for i in eachindex(σ2)
        if i/length(σ2) <= 2/3
            σ2[i] = 1.0
        end
    end

    for j in axes(u, 2), i in axes(u, 1)
        u[i, j] *= σ1[i] * σ2[j]
    end

    return nothing
end

begin
    x0 = 0
    x1 = 1
    ncell = 100
    nface = ncell + 1
    dx = (x1 - x0) / ncell
    deg = 2 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.05
    dt = cfl * dx
    t = 0.0
end

ps = FRPSpace1D(x0, x1, ncell, deg)

begin
    uqMethod = "galerkin"
    nr = 9
    nRec = 18
    opType = "uniform"
    parameter1 = 0.95
    parameter2 = 1.05
end

uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)

u = zeros(ncell, nsp, 3, uq.nm+1)
# stochastic density
#=for i = 1:ncell, j = 1:nsp
    prim = zeros(3, uq.nm+1)
    if ps.x[i] <= 0.5
        prim[1, :] .= uq.pce
        prim[2, 1] = 0.0
        prim[3, 1] = 0.5
    else
        prim[:, 1] .= [0.3, 0.0, 0.625]
    end

    u[i, j, :, :] .= uq_prim_conserve(prim, γ, uq)
end=#

# stochastic location
for i = 1:ncell, j = 1:nsp
    prim = zeros(3, uq.nq)

    for k = 1:uq.nq
        if ps.x[i] <= 0.5 + 0.05 * uq.op.quad.nodes[k]
            prim[:, k] .= [1.0, 0.0, 0.5]
        else
            prim[:, k] .= [0.4, 0.0, 0.625]
        end
    end

    prim_chaos = zeros(3, uq.nm+1)
    for k = 1:3
        prim_chaos[k, :] .= ran_chaos(prim[k, :], uq)
    end

    u[i, j, :, :] .= uq_prim_conserve(prim_chaos, γ, uq)
end

tspan = (0.0, 0.15)
p = (ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, γ, uq)

begin
    V = vandermonde_matrix(ps.deg,ps.xpl)
    VInv = inv(V)

    Nx = ncell
    nLocal = deg+1
    nStates = 3
    nMoments = uq.nm+1 # number of moments

    lambdaX = 1e-2
    lambdaXi = 1e-5
    filterType = "Lasso"

    # compute L1 norms of basis
    NxHat = 100
    xHat = collect(range(-1,stop = 1,length = NxHat))
    dxHat = xHat[2]-xHat[1];
    PhiL1 = zeros(nLocal,nMoments)
    for i = 1:nLocal
        for j = 1:nMoments
            PhiL1[i,j] = dxHat^2*sum(abs.(JacobiP(xHat, 0, 0, i-1).*JacobiP(xHat, 0, 0, j-1)))
        end
    end
end

u1 = deepcopy(u)
#=for j = 1:size(u,1)
    for s = 1:size(u,3)
        uModal = VInv*u[j,:,s,:]                
        u1[j,:,s,:] .= V*Filter(uModal,lambdaX,lambdaXi,filterType,PhiL1)
    end
end=#

prob = ODEProblem(dudt!, u1, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)

    # filter
    for j = 1:size(itg.u,1)
        for s = 1:size(itg.u,3)
            uModal = VInv * itg.u[j, :, s, :]
            modal_filter!(uModal, 10, 10; filter = :houli)
            itg.u[j, :, s, :] .= V * uModal
        end
    end
end

sol = zeros(ncell, nsp, 3, 2)
for i in axes(sol, 1), j in axes(sol, 2)
    p1 = zeros(3, uq.nm+1)
    p1 = uq_conserve_prim(itg.u[i, j, :, :], γ, uq)
    p1[end, :] .= lambda_tchaos(p1[end, :], 1.0, uq)

    for k = 1:3
        sol[i, j, k, 1] = mean(p1[k, :], uq.op)
        sol[i, j, k, 2] = std(p1[k, :], uq.op)
    end
end

pic1 = plot(ps.x, sol[:, 2, 1, 1], label="mean", xlabel="x", ylabel="ρ")
pic2 = plot(ps.x, sol[:, 2, 1, 2], label="std")
plot(pic1, pic2)
