#=function FR.filter_exp!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
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
end=#

function FR.filter_exp!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    nx, nz = size(u)
    λ = args[1]

    if length(args) >= 2
        Nc = args[2]
    else
        Nc = 0
    end

    σ = filter_exp2d(nx-1, nz-1, λ, Nc)

    for j in axes(u, 2), i in axes(u, 1)
        u[i, j] *= σ[i, j]
    end

    return nothing
end

function filter_exp2d(Nx, Ny, sp, Nc = 0)
    alpha = -log(eps())

    filterdiag = ones((Nx + 1), (Ny + 1))
    for i = 0:Nx
        for j = 0:Ny
            if i + j >= Nc
                filterdiag[i+1, j+1] = exp(-alpha * ((i + j - Nc) / (Nx + Ny - Nc))^sp)
            end
        end
    end

    return filterdiag
end

function FR.filter_houli!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
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

begin # compute L1 norms of basis
    Nx = ps.nx
    nLocal = ps.deg + 1
    nMoments = uq.nm + 1

    NxHat = 100
    xHat = collect(range(-1,stop = 1,length = NxHat))
    dxHat = xHat[2]-xHat[1];
    PhiL1 = zeros(nLocal,nMoments)
    for i = 1:nLocal
        for j = 1:nMoments
            PhiJ = sum(@. abs(uq.op.quad.weights * uq.phiRan[:, j])) / (uq.t2Product[j-1, j-1] + 1.e-7)
            PhiI = dxHat*sum(abs.(JacobiP(xHat, 0, 0, i-1)))
            PhiL1[i,j] = PhiI*PhiJ
        end
    end
end
