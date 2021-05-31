using Kinetic, ProgressMeter, Plots, LinearAlgebra, JLD2, Flux
using Flux: onecold, @epochs

###
# define neural model
###

cd(@__DIR__)
@load "data.jld2" X1 Y1 X2 Y2

device = gpu

X1 = Float32.(X1) |> device
Y1 = Float32.(Y1) |> device
X2 = Float32.(X2) |> device
Y2 = Float32.(Y2) |> device

nn = Chain(
    Dense(7, 28, relu),
    Dense(28, 56, relu),
    Dense(56, 28, relu),
    Dense(28, 2),
) |> device

data = Flux.Data.DataLoader((X1, Y1), shuffle = true) |> device
ps = params(nn)
sqnorm(x) = sum(abs2, x)
loss(x, y) = sum(abs2, nn(x) - y) / size(x, 2) #+ 1e-6 * sum(sqnorm, ps)
cb = () -> println("loss: $(loss(X1, Y1))")
opt = ADAM()

@epochs 5 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

#sci_train!(nn, (X1, Y1), ADAM(); device = gpu, epoch = 10)
#sci_train!(nn, (X1, Y1), ADAM(); device = cpu, epoch = 1)

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

nn = nn |> cpu

@save "nn.jld2" nn

nn(Array(X1[:, end]))
