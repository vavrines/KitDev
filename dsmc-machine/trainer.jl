using Kinetic, DataFrames, Flux, LinearAlgebra, JLD2, Kinetic.KitML.DiffEqFlux

isNewStart = true#false
cd(@__DIR__)

# read dsmc data
dfs = []
for i = 1:200
    df = DataFrame(idx = Int32[], t = Float32[], u = Float32[], v = Float32[], x = Float32[], y = Float32[])
    fname = "Track_particle/GroupPartTrack_iter(" * string(i) * ").dat"

    io = open(fname)
    for line in eachline(io)
        if length(line) == 0 || line[1] == 'G'
            continue
        end

        res = split(line, " "; keepempty = false)
        v0 = parse(Int32, res[1])
        v1, v2, v3, v4, v5 = parse.(Float32, res[2:end])

        push!(df, (v0, v1, v2, v3, v4, v5))
    end
    close(io)

    push!(dfs, df)
end

# generate dataset
X = [dfs[1].t; dfs[1].u; dfs[1].v]
Y = [dfs[2].t; dfs[2].u; dfs[2].v]

for i = 2:length(dfs)-1
    _X = [dfs[i].t; dfs[i].u; dfs[i].v]
    _Y = [dfs[i+1].t; dfs[i+1].u; dfs[i+1].v]
    global X = hcat(X, _X)
    global Y = hcat(Y, _Y)
end

# neural model
if isNewStart
    nn = Chain(
        Dense(9, 36, tanh),
        Dense(36, 72, tanh),
        Dense(72, 72, tanh),
        Dense(72, 36, tanh),
        Dense(36, 9),
    )
else
    @load "model.jld2" nn
end

# train
sci_train!(nn, (X, Y), ADAM(); device = cpu, batch = 16, epoch = 5000)

# accuracy
function accuracy()
    ac = 0
    pred = nn(X)
    for i in eachindex(X)
        if abs(pred[i] - Y[i]) / Y[i] <= 0.05
            ac += 1
        end
    end

    return ac / length(X)
end

@show accuracy()

# save model
@save "model.jld2" nn
