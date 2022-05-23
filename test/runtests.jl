using CrossValidation
import CrossValidation: predict, score

struct MyModel
    a::Int
    b::Int
end

function mymodel(x::AbstractArray; a::Int, b::Int)
    return MyModel(a, b)
end

function predict(model::MyModel, x::AbstractArray)
    return rand(size(x)...)
end

function score(model::MyModel, x::AbstractArray)
    return sum(predict(model, x) - x)
end

x = rand(2, 10)

search = ExhaustiveSearch(a=1:2, b=3:4)

cv = crossvalidate(mymodel, KFold(x), search)

cv = crossvalidate((x) -> mymodel(x, a=1, b=2), LeavePOut(x))

cv = crossvalidate((x) -> crossvalidate(mymodel, FixedSplit(x), search), RandomSplit(x))