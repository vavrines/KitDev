using KitBase, FluxReconstruction, OrdinaryDiffEq, Langevin, LinearAlgebra, Plots
using ProgressMeter: @showprogress

function Filter(u::Array{Float64,1},lambda::Float64,filterType::String)
    if filterType == "L2"
        for i = 1:length(u)
            u[i] = u[i]/(1.0+lambda*i^2*(i-1)^2);
        end
    elseif filterType == "Lasso"
        N = obj.settings.N;
        lambdaFilter = abs(u[N])/(N*(N-1)*PhiL1[N]);
        for i = 2:length(u)
            scL1 = 1.0-lambdaFilter*i*(i-1)*PhiL1[i]/abs(u[i]);
            if scL1 < 0 || abs(u[i]) < 1.0e-7
                scL1 = 0;
            end
            u[i] = scL1*u[i];
        end
    end
    return u;
end

function Filter(u::Array{Float64,2},lambdaX::Float64,lambdaXi::Float64,filterType::String,PhiL1=0.0)
    if filterType == "L2"
        for i = 1:size(u,1)
            for j = 1:size(u,2)
                u[i,j] = u[i,j]/(1.0+lambdaX*i^2*(i-1)^2+lambdaXi*j^2*(j-1)^2);
            end
        end
    elseif filterType == "Lasso"
        #PhiL1 = ones(size(u))
        N1 = size(u,1);
        N2 = size(u,2)
        lambda1 = abs(u[N1,1])/(N1*(N1-1)*PhiL1[N1,1]);
        lambda2 = abs(u[1,N2])/(N2*(N2-1)*PhiL1[1,N2]);
        for i = 1:size(u,1)
            for j = 1:size(u,2)
                scL1 = 1.0-lambda1*i*(i-1)*PhiL1[i,j]/abs(u[i,j])-lambda2*j*(j-1)*PhiL1[i,j]/abs(u[i,j]);
                if scL1 < 0 || abs(u[i,j]) < 1.0e-7
                    scL1 = 0;
                end
                u[i,j] = scL1*u[i,j];
            end
        end
    end
    return u;
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
for i = 1:ncell, j = 1:nsp
    prim = zeros(3, uq.nm+1)
    if ps.x[i] <= 0.5
        prim[1, :] .= uq.pce
        prim[2, 1] = 0.0
        prim[3, 1] = 0.5
    else
        prim[:, 1] .= [0.3, 0.0, 0.625]
    end

    u[i, j, :, :] .= uq_prim_conserve(prim, γ, uq)
end

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

function dudt!(du, u, p, t)
    du .= 0.0
    J, ll, lr, lpdm, dgl, dgr, γ, uq = p

    nm = uq.nm
    nq = uq.nq

    ncell = size(u, 1)
    nsp = size(u, 2)

    u_ran = zeros(ncell, nsp, 3, nq)
    for i = 1:ncell, j = 1:nsp, k = 1:3
        u_ran[i, j, k, :] .= chaos_ran(u[i, j, k, :], uq)
    end

    f = zeros(ncell, nsp, 3, nm+1)
    for i = 1:ncell, j = 1:nsp
        _f = zeros(3, nq)
        for k = 1:nq
            _f[:, k] .= euler_flux(u_ran[i, j, :, k], γ)[1] ./ J[i]
        end

        for k = 1:3
            f[i, j, k, :] .= ran_chaos(_f[k, :], uq)
        end
    end

    f_face = zeros(ncell, 3, nm+1, 2)
    for i = 1:ncell, j = 1:3, k = 1:nm+1
        f_face[i, j, k, 1] = dot(f[i, :, j, k], lr)
        f_face[i, j, k, 2] = dot(f[i, :, j, k], ll)
    end

    u_face = zeros(ncell, 3, nq, 2)
    for i = 1:ncell, j = 1:3, k = 1:nq
        u_face[i, j, k, 1] = dot(u_ran[i, :, j, k], lr)
        u_face[i, j, k, 2] = dot(u_ran[i, :, j, k], ll)
    end

    fq_interaction = zeros(ncell + 1, 3, nq)
    for i = 2:ncell, j = 1:nq
        fw = @view fq_interaction[i, :, j]
        flux_hll!(fw, u_face[i-1, :, j, 1], u_face[i, :, j, 2], γ, 1.0)
    end

    f_interaction = zeros(ncell + 1, 3, nm+1)
    for i = 2:ncell, j = 1:3
        f_interaction[i, j, :] .= ran_chaos(fq_interaction[i, j, :], uq)
    end

    rhs1 = zero(u)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:3, l = 1:nm+1
        rhs1[i, ppp1, k, l] = dot(f[i, :, k, l], lpdm[ppp1, :])
    end

    idx = 2:ncell-1 # ending points are Dirichlet
    for i in idx, ppp1 = 1:nsp, k = 1:3, l = 1:nm+1
        du[i, ppp1, k, l] =
            -(
                rhs1[i, ppp1, k, l] +
                (f_interaction[i, k, l] / J[i] - f_face[i, k, l, 2]) * dgl[ppp1] +
                (f_interaction[i+1, k, l] / J[i] - f_face[i, k, l, 1]) * dgr[ppp1]
            )
    end
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
            uModal = VInv*itg.u[j,:,s,:]                
            itg.u[j,:,s,:] .= V*Filter(uModal,lambdaX,lambdaXi,filterType,PhiL1)
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

cd(@__DIR__)
savefig("rho.png") 
