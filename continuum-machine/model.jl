using Kinetic, ProgressMeter, Plots, LinearAlgebra, JLD2
using Flux
using Flux: onecold, @epochs

###
# define neural model
###

cd(@__DIR__)
@load "data.jld2" X1 Y1 X2 Y2

nn = Chain(
    Dense(7, 14, relu),
    Dense(14, 28, relu),
    Dense(28, 14, relu),
    Dense(14, 2),
)

data = Flux.Data.DataLoader((X1, Y1), shuffle = true)
ps = params(nn)
sqnorm(x) = sum(abs2, x)
loss(x, y) = sum(abs2, nn(x) - y) / size(x, 2) #+ 1e-6 * sum(sqnorm, ps)
cb = () -> println("loss: $(loss(X1, Y1))")
opt = ADAM()

Flux.@epochs 10 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

#sci_train!(nn, (X, Y), ADAM(); device = cpu, epoch = 50)

# test
i = 50
sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
x, y = regime_data(ks, ctr[i].w, ctr[i].prim, sw, ctr[i].f)
nn(x) #|> onecold

Mu, Mxi, _, _1 = gauss_moments(ctr[i].prim, ks.gas.K)
a = pdf_slope(ctr[i].prim, sw, ks.gas.K)
sw = -ctr[i].prim[1] .* moments_conserve_slope(a, Mu, Mxi, 1)
A = pdf_slope(ctr[i].prim, sw, ks.gas.K)
fr = chapman_enskog(ks.vs.u, ctr[i].prim, a, A, tau)

plot(ks.vs.u, ctr[i].f)
plot!(ks.vs.u, fr, line=:dash)

# accuracy
function accuracy(X, Y)
    Y1 = nn(X)

    YA1 = [onecold(Y1[:, i]) for i in axes(Y1, 2)]
    YA = [onecold(Y[:, i]) for i in axes(Y, 2)]

    accuracy = 0.0
    for i in eachindex(YA)
        if YA[i] == YA1[i]
            accuracy += 1.0
        end
    end
    accuracy /= length(YA)

    return accuracy
end

accuracy(X1, Y1)
accuracy(X2, Y2)

@save "nn.jld2" nn
