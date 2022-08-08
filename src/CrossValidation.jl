module CrossValidation

using Base: @propagate_inbounds, Iterators.ProductIterator
using Random: shuffle!
using Distributed: pmap

export ResampleMethod, FixedSplit, RandomSplit, StratifiedSplit, KFold, StratifiedKFold, ForwardChaining, SlidingWindow,
       ExhaustiveSearch,
       ModelValidation, ParameterTuning, crossvalidate,
       predict, score

nobs(x::AbstractArray) = size(x)[end]

function nobs(x::Union{Tuple, NamedTuple})
    equalobs(x) || throw(ArgumentError("all data should have the same number of observations"))
    return nobs(x[1])
end

function equalobs(x::Union{Tuple, NamedTuple})
    length(x) > 0 || return false
    n = nobs(x[1])
    return all(y -> nobs(y) == n, Base.tail(x))
end

getobs(x::AbstractArray, i) = x[Base.setindex(map(Base.Slice, axes(x)), i, ndims(x))...]
getobs(x::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(getobs, i), x)

restype(x::Tuple) = Tuple{map(restype, x)...}
restype(x::NamedTuple) = NamedTuple{keys(x), Tuple{map(restype, x)...}}
restype(x::AbstractArray) = Array{eltype(x), ndims(x)}
restype(x::Any) = typeof(x)

abstract type ResampleMethod end

Base.eltype(r::ResampleMethod) = Tuple{restype(r.data), restype(r.data)}

struct FixedSplit{D} <: ResampleMethod
    data::D
    nobs::Int
    ratio::Number
end

function FixedSplit(data::Union{AbstractArray, Tuple, NamedTuple}, ratio::Number = 0.8)
    n = nobs(data)
    1 ≤ n * ratio ≤ n - 1 || throw(ArgumentError("data cannot be split based on a $ratio ratio"))
    return FixedSplit(data, n, ratio)
end

function FixedSplit(data::Union{AbstractArray, Tuple, NamedTuple}, m::Int)
    n = nobs(data)
    1 ≤ m ≤ n - 1 || throw(ArgumentError("data cannot be split by $m"))
    return FixedSplit(data, n, m / n)
end

Base.length(r::FixedSplit) = 1

@propagate_inbounds function Base.iterate(r::FixedSplit, state = 1)
    state > 1 && return nothing
    m = ceil(Int, r.ratio * r.nobs)
    train = getobs(r.data, 1:m)
    test = getobs(r.data, (m + 1):r.nobs)
    return (train, test), state + 1
end

struct RandomSplit{D} <: ResampleMethod
    data::D
    nobs::Int
    ratio::Number
    perm::Vector{Int}
end

function RandomSplit(data::Union{AbstractArray, Tuple, NamedTuple}, ratio::Number = 0.8)
    n = nobs(data)
    1 ≤ n * ratio ≤ n - 1 || throw(ArgumentError("data cannot be split based on a $ratio ratio"))
    return RandomSplit(data, n, ratio, shuffle!([1:n;]))
end

function RandomSplit(data::Union{AbstractArray, Tuple, NamedTuple}, m::Int)
    n = nobs(data)
    1 ≤ m ≤ n - 1 || throw(ArgumentError("data cannot be split by $m"))
    return RandomSplit(data, n, m / n, shuffle!([1:n;]))
end

Base.length(r::RandomSplit) = 1

@propagate_inbounds function Base.iterate(r::RandomSplit, state = 1)
    state > 1 && return nothing
    m = ceil(Int, r.ratio * r.nobs)
    train = getobs(r.data, r.perm[1:m])
    test = getobs(r.data, r.perm[(m + 1):r.nobs])
    return (train, test), state + 1
end

struct KFold{D} <: ResampleMethod
    data::D
    nobs::Int
    folds::Int
    perm::Vector{Int}
end

function KFold(data::Union{AbstractArray, Tuple, NamedTuple}; k::Int = 10)
    n = nobs(data)
    1 < k ≤ n || throw(ArgumentError("data cannot be partitioned into $k folds"))
    return KFold(data, n, k, shuffle!([1:n;]))
end

Base.length(r::KFold) = r.folds

@propagate_inbounds function Base.iterate(r::KFold, state = 1)
    state > r.folds && return nothing
    m = mod(r.nobs, r.folds)
    w = floor(Int, r.nobs / r.folds)
    fold = ((state - 1) * w + min(m, state - 1) + 1):(state * w + min(m, state))
    train = getobs(r.data, r.perm[1:end .∉ Ref(fold)])
    test = getobs(r.data, r.perm[fold])
    return (train, test), state + 1
end

struct ForwardChaining{D} <: ResampleMethod
    data::D
    nobs::Int
    init::Int
    out::Int
    partial::Bool
end

function ForwardChaining(data::Union{AbstractArray, Tuple, NamedTuple}, init::Int, out::Int; partial::Bool = true)
    n = nobs(data)
    1 ≤ init ≤ n || throw(ArgumentError("invalid initial window of $init"))
    1 ≤ out ≤ n || throw(ArgumentError("invalid out-of-sample window of $out"))
    init + out ≤ n || throw(ArgumentError("initial and out-of-sample window exceed the number of data observations"))
    return ForwardChaining(data, n, init, out, partial)
end

function Base.length(r::ForwardChaining)
    l = (r.nobs - r.init) / r.out
    return r.partial ? ceil(Int, l) : floor(Int, l)
end

@propagate_inbounds function Base.iterate(r::ForwardChaining, state = 1)
    state > length(r) && return nothing
    train = getobs(r.data, 1:(r.init + (state - 1) * r.out))
    test = getobs(r.data, (r.init + (state - 1) * r.out + 1):min(r.init + state * r.out, r.nobs))
    return (train, test), state + 1
end

struct SlidingWindow{D} <: ResampleMethod
    data::D
    nobs::Int
    window::Int
    out::Int
    partial::Bool
end

function SlidingWindow(data::Union{AbstractArray, Tuple, NamedTuple}, window::Int, out::Int; partial::Bool = true)
    n = nobs(data)
    1 ≤ window ≤ n || throw(ArgumentError("invalid sliding window of $window"))
    1 ≤ out ≤ n || throw(ArgumentError("invalid out-of-sample window of $out"))
    window + out ≤ n || throw(ArgumentError("sliding and out-of-sample window exceed the number of data observations"))
    return SlidingWindow(data, n, window, out, partial)
end

function Base.length(r::SlidingWindow)
    l = (r.nobs - r.window) / r.out
    return r.partial ? ceil(Int, l) : floor(Int, l)
end

@propagate_inbounds function Base.iterate(r::SlidingWindow, state = 1)
    state > length(r) && return nothing
    train = getobs(r.data, (1 + (state - 1) * r.out):(r.window + (state - 1) * r.out))
    test = getobs(r.data, (r.window + (state - 1) * r.out + 1):min(r.window + state * r.out, r.nobs))
    return (train, test), state + 1
end

struct ExhaustiveSearch
    keys::Tuple
    iter::ProductIterator
end

function ExhaustiveSearch(; args...)
    return ExhaustiveSearch(keys(args), Base.product(values(args)...))
end

Base.length(s::ExhaustiveSearch) = length(s.iter)
Base.eltype(s::ExhaustiveSearch) = NamedTuple{s.keys, eltype(s.iter)}

function Base.iterate(s::ExhaustiveSearch)
    (item, state) = iterate(s.iter)
    return (NamedTuple{s.keys}(item), state)
end

function Base.iterate(s::ExhaustiveSearch, state)
    next = iterate(s.iter, state)
    next === nothing && return nothing
    return (NamedTuple{s.keys}(next[1]), next[2])
end

_fit(x::AbstractArray, fit) = fit(x)
_fit(x::Union{Tuple, NamedTuple}, fit) = fit(x...)
_fit(x::AbstractArray, fit, args) = fit(x; args...)
_fit(x::Union{Tuple, NamedTuple}, fit, args) = fit(x...; args...)

_score(x::AbstractArray, model) = score(model, x)
_score(x::Union{Tuple, NamedTuple}, model) = score(model, x...)

struct ModelValidation{T1,T2}
    model::Vector{T1}
    score::Vector{T2}
end

struct ParameterSearch{T1,T2}
    model::Matrix{T1}
    score::Matrix{T2}
    final::T1
end

nopreprocess(train) = train
nopreprocess(train, test) = train, test

function crossvalidate(fit::Function, resample::ResampleMethod; preprocess::Function = nopreprocess, verbose::Bool = false)
    n = length(resample)
    model = Vector{Any}(undef, n)
    score = Vector{Any}(undef, n)

    i = 1
    for (train, test) in resample
        train, test = preprocess(train, test)

        model[i] = _fit(train, fit)
        score[i] = _score(test, model[i])

        if i == 1
            model = convert(Vector{typeof(model[1])}, model)
            score = convert(Vector{typeof(score[1])}, score)
        end

        if verbose
            @info "Completed iteration $i of $n"
        end

        i = i + 1
    end

    return ModelValidation(model, score)
end

function crossvalidate(fit::Function, resample::ResampleMethod, search::ExhaustiveSearch; preprocess::Function = nopreprocess, maximize::Bool = true, verbose::Bool = false)
    grid = collect(search)
    n, m = length(resample), length(grid)
    model = Matrix{Any}(undef, n, m)
    score = Matrix{Any}(undef, n, m)

    i = 1
    for (train, test) in resample
        train, test = preprocess(train, test)

        model[i,:] = pmap((args) -> _fit(train, fit, args), grid)
        score[i,:] = map((model) -> _score(test, model), model[i,:])

        if i == 1
            model = convert(Array{typeof(model[1])}, model)
            score = convert(Array{typeof(score[1])}, score)
        end

        if verbose
            if maximize
                best = max(score[i,:]...)
            else
                best = min(score[i,:]...)
            end
            @info "Completed iteration $i of $n"
        end

        i = i + 1
    end

    if maximize
        idx = argmax(sum(score, dims=1) ./ n)[2]
    else
        idx = argmin(sum(score, dims=1) ./ n)[2]
    end

    final = _fit(preprocess(resample.data), fit, grid[idx])

    if verbose
        @info "Completed fitting final model"
    end

    return ParameterSearch(model, score, final)
end

function predict(cv::ParameterSearch, kwargs...)
    return predict(cv.final, kwargs...)
end

function score(cv::ParameterSearch, kwargs...)
    return score(cv.final, kwargs...)
end

end
