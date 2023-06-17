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

sp = space(a = DiscreteUniform(-8.0:1.0:8.0), b = DiscreteUniform(-8.0:1.0:8.0))

brute(MyModel, sp, FixedSplit(x), false, epochs = 100)
brute(MyModel, sample(sp, 64), FixedSplit(x), false, epochs = 100)
hc(MyModel, sp, FixedSplit(x), 1, 1, false, epochs = 100)

sha(MyModel, sp, FixedSplit(x), Budget(epochs = 448), GeometricSchedule, 2, false)
sha(MyModel, sample(sp, 64), FixedSplit(x), Budget(epochs = 600), ConstantSchedule, 2, false)

hyperband(MyModel, sp, FixedSplit(x), Budget(epochs = 81), 3, false)

sasha(MyModel, sp, FixedSplit(x), 1, false, epochs = 1)

validate(KFold(x)) do train
    prms = brute(MyModel, sp, FixedSplit(train), false, epochs = 100)
    return fit!(MyModel(prms...), train, epochs = 10)
end

validate(KFold(x)) do train
    prms = hc(MyModel, sp, FixedSplit(train), 1, 1, false, epochs = 100)
    return fit!(MyModel(prms...), train, epochs = 10)
end

validate(KFold(x)) do train
    prms = sha(MyModel, sp, FixedSplit(train), Budget(epochs = 100), GeometricSchedule, 2, false)
    return fit!(MyModel(prms...), train, epochs = 10)
end

validate(KFold(x)) do train
    prms = sasha(MyModel, sp, FixedSplit(train), 1, false, epochs = 1)
    return fit!(MyModel(prms...), train, epochs = 10)
end
