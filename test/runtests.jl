using CrossValidation

import CrossValidation: loss, fit!

struct MyModel
    a::Int
    b::Float64
end

function mymodel(x::AbstractArray; a::Int, b::Float64)
    return MyModel(a, b)
end

function fit!(model::MyModel, x::AbstractArray; epochs::Int = 1)
    return model
end

function loss(model::MyModel, x::AbstractArray)
    return model.a^2 + model.b^2
end

x = rand(2, 100)

scores = cv(x -> mymodel(x, a=1, b=2.0), FixedSplit(x))
scores = cv(x -> mymodel(x, a=1, b=2.0), RandomSplit(x))
scores = cv(x -> mymodel(x, a=1, b=2.0), KFold(x))
scores = cv(x -> mymodel(x, a=1, b=2.0), ForwardChaining(x, 40, 10))
scores = cv(x -> mymodel(x, a=1, b=2.0), SlidingWindow(x, 40, 10))

space = ParameterSpace{(:a, :b)}(-10:10, -10.0:0.5:10.0)

model = brute(mymodel, GridSampler(space), FixedSplit(x))
model = brute(mymodel, RandomSampler(space, n=100), FixedSplit(x))
model = hc(mymodel, space, FixedSplit(x))

budget = (epochs = 10000,)

model = sha(MyModel, GridSampler(space), budget, FixedSplit(x))
model = sha(MyModel, RandomSampler(space, n=100), budget, FixedSplit(x))

f(train) = train ./ 10
f(train, test) = train ./ 10, test ./ 10

scores = cv(x -> mymodel(x, a=1, b=2.0), PreProcess(FixedSplit(x), f))
model = brute(mymodel, RandomSampler(space, n=100), PreProcess(KFold(x), f))

scores = cv(x -> brute(mymodel, GridSampler(space), FixedSplit(x)), KFold(x))
scores = cv(x -> hc(mymodel, space, FixedSplit(x)), KFold(x))
scores = cv(x -> sha(MyModel, GridSampler(space), budget, FixedSplit(x)), KFold(x))
