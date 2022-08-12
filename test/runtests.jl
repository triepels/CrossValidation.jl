using CrossValidation

import CrossValidation: loss

struct MyModel
    a::Int
    b::Float64
end

function mymodel(x::AbstractArray; a::Int, b::Float64)
    return MyModel(a, b)
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

space = SearchSpace{(:a, :b)}(-10:10, -10.0:0.5:10.0)

model = optimize(mymodel, ExhaustiveSearch(space), FixedSplit(x))
model = optimize(mymodel, RandomSearch(space, 100), FixedSplit(x))

f(train) = train ./ 10
f(train, test) = train ./ 10, test ./ 10

scores = cv(x -> mymodel(x, a=1, b=2.0), PreProcess(FixedSplit(x), f))
model = optimize(mymodel, RandomSearch(space, 100), PreProcess(KFold(x), f))

scores = cv(x -> optimize(mymodel, ExhaustiveSearch(space), FixedSplit(x)), KFold(x))
