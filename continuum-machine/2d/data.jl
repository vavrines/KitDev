using Kinetic, Plots, LinearAlgebra, JLD2
using ProgressMeter: @showprogress

X = Float32.([[1.0, 0.0, 0.0, 1.0]; zeros(4); 1e-4])
Y = Float32.([1.0, 0.0])

function init_field!(ks, ctr, a1face, a2face)
    for i in eachindex(ctr)
        prim = [2.0 * rand(), 0.0, 0.0, 1 / rand()]
    
        ctr[i].prim .= prim
        ctr[i].w .= prim_conserve(prim, ks.gas.γ)
        ctr[i].f .= maxwellian(ks.vs.u, ks.vs.v, prim)
    end
    for i in eachindex(a1face)
        a1face[i].fw .= 0.0
        a1face[i].ff .= 0.0
    end
    for i in eachindex(a2face)
        a2face[i].fw .= 0.0
        a2face[i].ff .= 0.0
    end
end

function regime_data(ks, w, prim, swx, swy, f)
    Mu, Mv, Mxi, _, _1 = gauss_moments(prim, ks.gas.K)
    a = pdf_slope(prim, swx, ks.gas.K)
    b = pdf_slope(prim, swy, ks.gas.K)
    sw = -prim[1] .* (moments_conserve_slope(a, Mu, Mv, Mxi, 1, 0) .+ moments_conserve_slope(b, Mu, Mv, Mxi, 0, 1))
    A = pdf_slope(prim, sw, ks.gas.K)
    tau = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
    
    fr = chapman_enskog(ks.vs.u, ks.vs.v, prim, a, b, A, tau)
    L = norm((f .- fr) ./ prim[1])

    sw = (swx.^2 + swy.^2).^0.5
    x = [w; sw; tau]
    y = ifelse(L <= 0.005, [1.0, 0.0], [0.0, 1.0])

    return x, y
end

cd(@__DIR__)
D = Dict{Symbol,Any}()
begin
    D[:matter] = "gas"
    D[:case] = "cavity"
    D[:space] = "2d1f2v"
    D[:flux] = "kfvs"
    D[:collision] = "bgk"
    D[:nSpecies] = 1
    D[:interpOrder] = 2
    D[:limiter] = "vanleer"
    D[:boundary] = "period"
    D[:cfl] = 0.5
    D[:maxTime] = 5.0

    D[:x0] = 0.0
    D[:x1] = 1.0
    D[:nx] = 41
    D[:y0] = 0.0
    D[:y1] = 1.0
    D[:ny] = 41
    D[:pMeshType] = "uniform"
    D[:nxg] = 1
    D[:nyg] = 1

    D[:umin] = -5.0
    D[:umax] = 5.0
    D[:nu] = 48
    D[:vmin] = -5.0
    D[:vmax] = 5.0
    D[:nv] = 48
    D[:vMeshType] = "rectangle"
    D[:nug] = 0
    D[:nvg] = 0

    D[:knudsen] = 0.05
    D[:mach] = 0.0
    D[:prandtl] = 1.0
    D[:inK] = 0.0
    D[:omega] = 0.81
    D[:alphaRef] = 1.0
    D[:omegaRef] = 0.5
end

for loop = 1:10
    println("iteration: $loop of 10")
    D[:knudsen] = exp(-Float64(loop))
    ks = SolverSet(D)
    ctr, a1face, a2face = init_fvm(ks, ks.ps, :dynamic_array; structarray = true)
    init_field!(ks, ctr, a1face, a2face)

    t = 0.0
    dt = timestep(ks, ctr, t)
    nt = Int(ks.set.maxTime ÷ dt) + 1
    res = zero(ks.ib.wL)
    @showprogress for iter = 1:1000#nt
        if iter%39 == 0
            for j = 1:ks.ps.ny, i = 1:ks.ps.nx
                swx = (ctr[i+1, j].w .- ctr[i-1, j].w) / ks.ps.dx[i, j] / 2.0
                swy = (ctr[i, j+1].w .- ctr[i, j-1].w) / ks.ps.dy[i, j] / 2.0
                x, y = regime_data(ks, ctr[i, j].w, ctr[i, j].prim, swx, swy, ctr[i, j].f)
                X = hcat(X, x)
                Y = hcat(Y, y)
            end
        end
        
        reconstruct!(ks, ctr)
        evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
        update!(ks, ctr, a1face, a2face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))

        t += dt
    end
end

function split_dataset(X, Y, ratio = 9::Integer)
    idx2 = rand(size(X, 2) ÷ ratio) * size(X, 2) .|> floor .|> Int
    sort!(idx2)
    unique!(idx2)
    idx1 = setdiff(collect(1:size(X, 2)), idx2)

    x_train = [X[:, j] for j in idx1]
    x_test = [X[:, j] for j in idx2]
    y_train = [Y[:, j] for j in idx1]
    y_test = [Y[:, j] for j in idx2]
    
    X_train = zeros(eltype(X), size(X, 1), length(x_train))
    Y_train = zeros(eltype(X), size(Y, 1), length(x_train))
    X_test = zeros(eltype(X), size(X, 1), length(x_test))
    Y_test = zeros(eltype(X), size(Y, 1), length(x_test))
    for j in axes(X_train, 2)
        X_train[:, j] .= x_train[j]
        Y_train[:, j] .= y_train[j]
    end
    for j in axes(X_test, 2)
        X_test[:, j] .= x_test[j]
        Y_test[:, j] .= y_test[j]
    end

    return X_train, Y_train, X_test, Y_test
end

X1, Y1, X2, Y2 = split_dataset(X, Y)

@save "data.jld2" X1 Y1 X2 Y2
