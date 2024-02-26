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

validate(MyModel(a = 2.0, b = 2.0), data, args = (epochs = 100,))

sp = space(a = DiscreteUniform(-8.0:1.0:8.0), b = DiscreteUniform(-8.0:1.0:8.0))

brute((x) -> MyModel(; x...), sp, data, args = (epochs = 100,), maximize = false)
brutefit((x) -> MyModel(; x...), sp, data, args = (epochs = 100,), maximize = false)

brute((x) -> MyModel(; x...), rand(sp, 64), data, args = (epochs = 100,), maximize = false)
brutefit((x) -> MyModel(; x...), rand(sp, 64), data, args = (epochs = 100,), maximize = false)

hc((x) -> MyModel(; x...), sp, data, 1, args = (epochs = 100,), n = 10, maximize = false)
hcfit((x) -> MyModel(; x...), sp, data, 1, args = (epochs = 100,), n = 10, maximize = false)

sha((x) -> MyModel(; x...), sp, data, Budget{:epochs}(448), mode = GeometricAllocation, rate = 2, maximize = false)
shafit((x) -> MyModel(; x...), sp, data, Budget{:epochs}(448), mode = GeometricAllocation, rate = 2, maximize = false)

sha((x) -> MyModel(; x...), rand(sp, 64), data, Budget{:epochs}(600), mode = ConstantAllocation, rate = 2, maximize = false)
shafit((x) -> MyModel(; x...), rand(sp, 64), data, Budget{:epochs}(600), mode = ConstantAllocation, rate = 2, maximize = false)

hyperband((x) -> MyModel(; x...), sp, data, Budget{:epochs}(81), rate = 3, maximize = false)
hyperbandfit((x) -> MyModel(; x...), sp, data, Budget{:epochs}(81), rate = 3, maximize = false)

sasha((x) -> MyModel(; x...), sp, data, args = (epochs = 1,), temp = 1, maximize = false)
sashafit((x) -> MyModel(; x...), sp, data, args = (epochs = 1,), temp = 1, maximize = false)

validate(KFold(x, 10)) do train
    parm = brute((x) -> MyModel(; x...), sp, FixedSplit(train, 0.8), args = (epochs = 100,), maximize = false)
    return fit!(MyModel(; parm...), train, epochs = 10)
end

validate(KFold(x, 10)) do train
    parm = hc((x) -> MyModel(; x...), sp, FixedSplit(train, 0.8), 1, args = (epochs = 100,), n = 10, maximize = false)
    return fit!(MyModel(; parm...), train, epochs = 10)
end

validate(KFold(x, 10)) do train
    parm = sha((x) -> MyModel(; x...), sp, FixedSplit(train, 0.8), Budget{:epochs}(100), mode = GeometricAllocation, rate = 2, maximize = false)
    return fit!(MyModel(; parm...), train, epochs = 10)
end

validate(KFold(x, 10)) do train
    parm = sasha((x) -> MyModel(; x...), sp, FixedSplit(train, 0.8), args = (epochs = 1,), temp = 1, maximize = false)
    return fit!(MyModel(; parm...), train, epochs = 10)
end

validate(KFold(x, 10)) do train
    return brutefit((x) -> MyModel(; x...), sp, FixedSplit(train, 0.8), args = (epochs = 100,), maximize = false)
end

validate(KFold(x, 10)) do train
    return hcfit((x) -> MyModel(; x...), sp, FixedSplit(train, 0.8), 1, args = (epochs = 100,), n = 10, maximize = false)
end

validate(KFold(x, 10)) do train
    return shafit((x) -> MyModel(; x...), sp, FixedSplit(train, 0.8), Budget{:epochs}(100), mode = GeometricAllocation, rate = 2, maximize = false)
end

validate(KFold(x, 10)) do train
    return sashafit((x) -> MyModel(; x...), sp, FixedSplit(train, 0.8), args = (epochs = 1,), temp = 1, maximize = false)
end
