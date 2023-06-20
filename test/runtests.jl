using CrossValidation

import CrossValidation: fit!, loss

struct MyModel
    a::Float64
    b::Float64
    MyModel(; a::Float64, b::Float64) = new(a, b)
end

function fit!(model::MyModel, x::AbstractArray; epochs::Int = 1)
    #println("Fitting $model ..."); sleep(0.1)
    return model
end

# Himmelblau's function
function loss(model::MyModel, x::AbstractArray)
    a, b = model.a, model.b
    return (a^2 + b - 11)^2 + (a + b^2 - 7)^2
end

collect(FixedSplit(1:10, 0.8))
collect(RandomSplit(1:10, 0.8))
collect(LeaveOneOut(1:10))
collect(KFold(1:10, 10))
collect(ForwardChaining(1:10, 4, 2))
collect(SlidingWindow(1:10, 4, 2))

x = rand(2, 100)
data = FixedSplit(x, 0.8)

validate(MyModel(a = 2.0, b = 2.0), data, args=(epochs = 100,))

sp = space(a = DiscreteUniform(-8.0:1.0:8.0), b = DiscreteUniform(-8.0:1.0:8.0))

brute(MyModel, sp, data, args = (epochs = 100,), maximize = false)
brute(MyModel, sample(sp, 64), data, args = (epochs = 100,), maximize = false)
hc(MyModel, sp, data, args = (epochs = 100,), nstart = 10, k = 1, maximize = false)

sha(MyModel, sp, data, Budget{:epochs}(448), mode = GeometricAllocation, rate = 2, maximize = false)
sha(MyModel, sample(sp, 64), data, Budget{:epochs}(600), mode = ConstantAllocation, rate = 2, maximize = false)

hyperband(MyModel, sp, data, Budget{:epochs}(81), rate = 3, maximize = false)

sasha(MyModel, sp, data, args = (epochs = 1,), temp = 1, maximize = false)

validate(KFold(x, 10)) do train
    parms = brute(MyModel, sp, FixedSplit(train, 0.8), args = (epochs = 100,), maximize = false)
    return fit!(MyModel(; parms...), train, epochs = 10)
end

validate(KFold(x, 10)) do train
    parms = hc(MyModel, sp, FixedSplit(train, 0.8), args = (epochs = 100,), nstart = 10, k = 1, maximize = false)
    return fit!(MyModel(; parms...), train, epochs = 10)
end

validate(KFold(x, 10)) do train
    parms = sha(MyModel, sp, FixedSplit(train, 0.8), Budget{:epochs}(100), mode = GeometricAllocation, rate = 2, maximize = false)
    return fit!(MyModel(; parms...), train, epochs = 10)
end

validate(KFold(x, 10)) do train
    parms = sasha(MyModel, sp, FixedSplit(train, 0.8), args = (epochs = 1,), temp = 1, maximize = false)
    return fit!(MyModel(; parms...), train, epochs = 10)
end
