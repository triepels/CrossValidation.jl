using CrossValidation

import CrossValidation: fit!, loss

struct MyModel
    a::Int
    b::Float64
end

function fit!(model::MyModel, x::AbstractArray; epochs::Int = 1)
    return model
end

function loss(model::MyModel, x::AbstractArray)
    return model.a^2 + model.b^2
end

x = rand(2, 100)

validate(MyModel(1, 2.0), (epochs=100,), FixedSplit(x))
validate(MyModel(1, 2.0), (epochs=100,), RandomSplit(x))
validate(MyModel(1, 2.0), (epochs=100,), KFold(x))
validate(MyModel(1, 2.0), (epochs=100,), ForwardChaining(x, 40, 10))
validate(MyModel(1, 2.0), (epochs=100,), SlidingWindow(x, 40, 10))

space = ParameterSpace(a = -10:10, b = -10.0:0.5:10.0)

model = brute(MyModel, GridSampler(space), (epochs=100,), FixedSplit(x))
model = brute(MyModel, RandomSampler(space, n=100), (epochs=100,), FixedSplit(x))
model = hc(MyModel, space, (epochs=100,), FixedSplit(x))

model = sha(MyModel, GridSampler(space), ConstantBudget((epochs=100,)), FixedSplit(x))
model = sha(MyModel, RandomSampler(space, n=100), GeometricBudget((epochs=100,), 1.5), FixedSplit(x))

f(train) = train ./ 10
f(train, test) = train ./ 10, test ./ 10

scores = validate(MyModel(1, 2.0), (epochs=100,), PreProcess(FixedSplit(x), f))
model = brute(MyModel, RandomSampler(space, n=100), (epochs=100,), PreProcess(KFold(x), f))

scores = validate(x -> brute(MyModel, GridSampler(space), (epochs=100,), FixedSplit(x)), KFold(x))
scores = validate(x -> hc(MyModel, space, (epochs=100,), FixedSplit(x)), KFold(x))
scores = validate(x -> sha(MyModel, GridSampler(space), ConstantBudget((epochs=100,)), FixedSplit(x)), KFold(x))
