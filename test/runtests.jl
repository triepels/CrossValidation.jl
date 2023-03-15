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

x = rand(2, 100)

validate(MyModel(2.0, 2.0), FixedSplit(x), epochs=100)
validate(MyModel(2.0, 2.0), RandomSplit(x), epochs=100)
validate(MyModel(2.0, 2.0), KFold(x), epochs=100)
validate(MyModel(2.0, 2.0), ForwardChaining(x, 40, 10), epochs=100)
validate(MyModel(2.0, 2.0), SlidingWindow(x, 40, 10), epochs=100)

space = Space(a = -6.0:0.5:6.0, b = -6.0:0.5:6.0)

brute(MyModel, space, FixedSplit(x), false, epochs=100)
brute(MyModel, sample(space, 100), FixedSplit(x), false, epochs=100)
hc(MyModel, space, FixedSplit(x), 1, false, epochs=100)

sha(MyModel, space, FixedSplit(x), ConstantBudget(epochs=100), 0.5, false)
sha(MyModel, sample(space, 100), FixedSplit(x), GeometricBudget(epochs=100), 0.5, false)

sasha(MyModel, space, FixedSplit(x), 1, false, epochs=1)

f(train) = train ./ 10
f(train, test) = train ./ 10, test ./ 10

validate(MyModel(2.0, 2.0), PreProcess(FixedSplit(x), f), epochs=100)
brute(MyModel, sample(space, 100), PreProcess(KFold(x), f), false, epochs=100)

validate(KFold(x)) do train
    parms = brute(MyModel, space, FixedSplit(train), false, epochs=100)
    return fit!(MyModel(parms...), train)
end

validate(KFold(x)) do train
    parms = hc(MyModel, space, FixedSplit(train), 1, false, epochs=100)
    return fit!(MyModel(parms...), train)
end

validate(KFold(x)) do train
    parms = sha(MyModel, space, FixedSplit(train), ConstantBudget(epochs=100), 0.5, false)
    return fit!(MyModel(parms...), train)
end

validate(KFold(x)) do train
    parms = sasha(MyModel, space, FixedSplit(train), 1, false, epochs=1)
    return fit!(MyModel(parms...), train)
end
