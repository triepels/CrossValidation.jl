using CrossValidation

import CrossValidation: fit!, loss

struct MyModel
    a::Float64
    b::Float64
end

MyModel(; a::Float64, b::Float64) = MyModel(a, b)

function fit!(model::MyModel, x::AbstractArray; epochs::Int = 1)
    #println("Fitting $model ..."); sleep(0.1)
    return model
end

# Himmelblau's function
function loss(model::MyModel, x::AbstractArray)
    a, b = model.a, model.b
    return (a^2 + b - 11)^2 + (a + b^2 - 7)^2
end

collect(FixedSplit(1:10))
collect(RandomSplit(1:10))
collect(LeaveOneOut(1:10))
collect(KFold(1:10))
collect(ForwardChaining(1:10, 4, 2))
collect(SlidingWindow(1:10, 4, 2))

x = rand(2, 100)

validate(MyModel(2.0, 2.0), FixedSplit(x), epochs = 100)

space = FiniteSpace(a = -8.0:1.0:8.0, b = -8.0:1.0:8.0)

brute(MyModel, space, FixedSplit(x), false, epochs = 100)
brute(MyModel, sample(space, 64), FixedSplit(x), false, epochs = 100)
hc(MyModel, space, FixedSplit(x), 1, 1, false, epochs = 100)

sha(MyModel, space, FixedSplit(x), ConstantBudget(epochs = 600), 2, false)
sha(MyModel, sample(space, 64), FixedSplit(x), GeometricBudget(epochs = 448), 2, false)

hyperband(MyModel, space, FixedSplit(x), HyperBudget(epochs = 81), 3, false)

sasha(MyModel, space, FixedSplit(x), 1, false, epochs = 1)

validate(KFold(x)) do train
    prms = brute(MyModel, space, FixedSplit(train), false, epochs = 100)
    return fit!(MyModel(prms...), train, epochs = 10)
end

validate(KFold(x)) do train
    prms = hc(MyModel, space, FixedSplit(train), 1, 1, false, epochs = 100)
    return fit!(MyModel(prms...), train, epochs = 10)
end

validate(KFold(x)) do train
    prms = sha(MyModel, space, FixedSplit(train), ConstantBudget(epochs = 100), 2, false)
    return fit!(MyModel(prms...), train, epochs = 10)
end

validate(KFold(x)) do train
    prms = sasha(MyModel, space, FixedSplit(train), 1, false, epochs = 1)
    return fit!(MyModel(prms...), train, epochs = 10)
end
