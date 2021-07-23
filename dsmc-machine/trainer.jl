using Kinetic, DataFrames, Flux, LinearAlgebra, JLD2

isNewStart = false
cd(@__DIR__)

# read dsmc data
dfs = []
for i = 1:50
    df = DataFrame(idx = Int32[], u = Float32[], v = Float32[], x = Float32[], y = Float32[])
    fname = "Track_particle/PartTrack_iter(" * string(i) * ").dat"

    io = open(fname)
    for line in eachline(io)
        if length(line) == 0 || line[1] == 'P'
            continue
        end

        res = split(line, " "; keepempty = false)
        v0 = parse(Int32, res[1])
        v1, v2, v3, v4 = parse.(Float32, res[2:end])

        push!(df, (v0, v1, v2, v3, v4))
    end
    close(io)

    push!(dfs, df)
end

# generate dataset
X = zeros(Float32, 800, 1)
Y = zeros(Float32, 800, 1)
X = [dfs[1].u; dfs[1].v]
Y = [dfs[2].u; dfs[2].v]

for i = 2:length(dfs)-1
    _X = [dfs[i].u; dfs[i].v]
    _Y = [dfs[i+1].u; dfs[i+1].v]
    global X = hcat(X, _X)
    global Y = hcat(Y, _Y)
end

# neural model
if isNewStart
    nn = Chain(Dense(800, 800, tanh), Dense(800, 800, tanh), Dense(800, 800))
else
    @load "model.jld2" nn
end

# train
sci_train!(nn, (X, Y), ADAM(); device = cpu, batch = 7, epoch = 100000)

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
