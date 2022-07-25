using Kinetic, ProgressMeter, Plots, LinearAlgebra, JLD2, Flux
using Flux: onecold, @epochs
using ProgressMeter: @showprogress

cd(@__DIR__)
@load "data.jld2" X1 Y1 X2 Y2

device = cpu

X1 = Float32.(X1) |> device
Y1 = Float32.(Y1) |> device
X2 = Float32.(X2) |> device
Y2 = Float32.(Y2) |> device

nn = Chain(
    RNN(7, 28, relu),
    #Dense(7, 28, relu),
    RNN(28, 56, relu),
    RNN(56, 28, relu),
    Dense(28, 2),
)

data = Flux.Data.DataLoader((X1, Y1), shuffle = true) |> device
ps = params(nn)
sqnorm(x) = sum(abs2, x)
loss(x, y) = sum(abs2, nn(x) - y) / size(x, 2) #+ 1e-6 * sum(sqnorm, ps)
cb = () -> println("loss: $(loss(X1, Y1))")
opt = ADAM()

for epoch = 1:10
    println("epoch: $epoch, loss: $(loss(X1, Y1))")
    gs = Flux.gradient(ps) do
        loss(X1, Y1)
    end
    Flux.Optimise.update!(opt, ps, gs)
end
