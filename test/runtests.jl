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
    return rand()
end

x = rand(2, 100)

scores = cv(x -> mymodel(x, a=1, b=2.0), FixedSplit(x))
scores = cv(x -> mymodel(x, a=1, b=2.0), RandomSplit(x))
scores = cv(x -> mymodel(x, a=1, b=2.0), KFold(x))
scores = cv(x -> mymodel(x, a=1, b=2.0), ForwardChaining(x, 40, 10))
scores = cv(x -> mymodel(x, a=1, b=2.0), SlidingWindow(x, 40, 10))

space = SearchSpace{(:a, :b)}(1:100, 1.0:0.1:10.0)

model = optimize(mymodel, FixedSplit(x), ExhaustiveSearch(space))
model = optimize(mymodel, FixedSplit(x), RandomSearch(space, 100))

nested = cv((x) -> optimize(mymodel, FixedSplit(x), ExhaustiveSearch(space)), RandomSplit(x))
