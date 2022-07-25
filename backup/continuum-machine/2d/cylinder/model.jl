using Kinetic, JLD2, Flux, Plots
using KitML.Solaris
using Flux: @epochs
using Flux: onecold

@load "../nn.jld2" nn
@load "kn3_aux.jld2" X Y

device = cpu

data = Flux.Data.DataLoader((X, Y), shuffle = true) |> device
ps = params(nn)
sqnorm(x) = sum(abs2, x)
loss(x, y) = sum(abs2, nn(x) - y) / size(x, 2) #+ 1e-6 * sum(sqnorm, ps)
cb = () -> println("loss: $(loss(X, Y))")
opt = ADAM()

@epochs 500 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

@load "kn3.jld2" ks ctr





cd(@__DIR__)
include("tools.jl")

rmap = zeros(ks.ps.nr, ks.ps.nθ)
@inbounds Threads.@threads for j = 1:ks.ps.nθ
    rmap[1, j] = 2
    for i = 2:ks.ps.nr
        swx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
        swy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
        swx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
        swy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.y[i, j+1] - ks.ps.y[i, j-1])
        swx = (swx1 + swx2) ./ 2
        swy = (swy1 + swy2) ./ 2
        #sw = sqrt.(sw1.^2 + sw2.^2)

        rmap[i, j] = judge_regime(ks, ctr[i, j].h, ctr[i, j].prim, swx, swy)
    end
end

@inbounds Threads.@threads for j = 1:ks.ps.nθ
    rmap[1, j] = 2
    for i = 2:ks.ps.nr
        swx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
        swy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
        swx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
        swy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.y[i, j+1] - ks.ps.y[i, j-1])
        swx = (swx1 + swx2) ./ 2
        swy = (swy1 + swy2) ./ 2
        sw = sqrt.(swx .^ 2 + swy .^ 2)
        tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)

        rmap[i, j] = nn([ctr[i, j].w; sw; tau]) |> onecold
    end
end

begin
    contourf(
        ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
        ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
        rmap[:, :],
        ratio = 1,
    )
end

nn(rand(9))
