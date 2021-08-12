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

function FR.filter_l2opt!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    p0 = axes(u, 1) |> first
    q0 = axes(u, 2) |> first
    @assert p0 >= 0
    @assert q0 >= 0

    λx, λξ = args[1:2]

    η0 = λx * 2^2 + λξ * 2^2
    for j in axes(u, 2)
        for i in axes(u, 1)
            if i == p0 && j == q0
                continue
            elseif i == 1
                η = λξ * 2^2
            elseif j == 1
                η = λx * 2^2
            else
                η = η0
            end

            u[i, j] /= (1.0 + λx * (i - p0 + 1)^2 * (i - p0)^2 + λξ * (j - q0 + 1)^2 * (j - q0)^2 - η)
        end
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

function FR.filter_lasso!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    nx, nz = size(u)
    
    λ1 = abs(u[nx,1])/(nx*(nx-1)*PhiL1[nx,1])
    λ2 = abs(u[1,nz])/(nz*(nz-1)*PhiL1[1,nz])

    for i = 1:nx
        for j = 1:nz
            scL1 = 1.0-(λ1*i*(i-1)+λ2*j*(j-1))*PhiL1[i,j]/abs(u[i,j])
            if scL1 < 0 || abs(u[i,j]) < eps()
                scL1 = 0
            end
            u[i, j] *= scL1
        end
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
