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

x = rand(2, 100)
y = rand(100) .≥ 0.5

search = ExhaustiveSearch(a=1:2, b=3:4)

cv = crossvalidate(mymodel, FixedSplit(x), search)
cv = crossvalidate(mymodel, RandomSplit(x), search)
cv = crossvalidate(mymodel, StratifiedSplit(x, y), search)
cv = crossvalidate(mymodel, KFold(x), search)
cv = crossvalidate(mymodel, StratifiedKFold(x, y), search)
cv = crossvalidate(mymodel, ForwardChaining(x, 40, 10), search)
cv = crossvalidate(mymodel, SlidingWindow(x, 40, 10), search)

cv = crossvalidate((x) -> crossvalidate(mymodel, FixedSplit(x), search), RandomSplit(x))