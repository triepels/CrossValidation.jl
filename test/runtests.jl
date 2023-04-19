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

space = DiscreteSpace(a = -6.0:0.5:6.0, b = -6.0:0.5:6.0)

brute(MyModel, space, FixedSplit(x), false, epochs = 100)
brute(MyModel, sample(space, 100), FixedSplit(x), false, epochs = 100)
hc(MyModel, space, FixedSplit(x), 1, 1, false, epochs = 100)

sha(MyModel, space, FixedSplit(x), ConstantBudget(0.5, epochs = 100), false)
sha(MyModel, sample(space, 100), FixedSplit(x), GeometricBudget(0.5, epochs = 100), false)

sasha(MyModel, space, FixedSplit(x), 1, false, epochs = 1)

validate(KFold(x)) do train
    prms = brute(MyModel, space, FixedSplit(train), false, epochs = 100)
    return fit!(MyModel(prms...), train)
end

validate(KFold(x)) do train
    prms = hc(MyModel, space, FixedSplit(train), 1, 1, false, epochs = 100)
    return fit!(MyModel(prms...), train)
end

validate(KFold(x)) do train
    prms = sha(MyModel, space, FixedSplit(train), ConstantBudget(0.5, epochs = 100), false)
    return fit!(MyModel(prms...), train)
end

validate(KFold(x)) do train
    prms, args = sasha(MyModel, space, FixedSplit(train), 1, false, epochs = 1)
    return fit!(MyModel(prms...), train; args...)
end
