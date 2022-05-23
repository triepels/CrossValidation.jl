module CrossValidation

using Base: @propagate_inbounds, Iterators.ProductIterator, product
using Random: shuffle!
using Distributed: pmap

export ResampleMethod, FixedSplit, RandomSplit, LeavePOut, KFold,
       ExhaustiveSearch,
       ModelValidation, ParameterTuning, crossvalidate,
       predict, score

# Based on Knet's src/data.jl
_nobs(data::AbstractArray) = size(data)[end]

function _nobs(data::Union{Tuple, NamedTuple})
    length(data) > 0 || throw(ArgumentError("Need at least one data input"))
    n = _nobs(data[1])
    if !all(x -> _nobs(x) == n, Base.tail(data))
        throw(DimensionMismatch("All data should contain same number of observations"))
    end
    return n
end

# Based on Knet's src/data.jl
_getobs(x::AbstractArray, i) = x[ntuple(i -> Colon(), Val(ndims(x) - 1))..., i]
_getobs(x::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(_getobs, i), x)

abstract type ResampleMethod end

struct FixedSplit{D} <: ResampleMethod
    data::D
    nobs::Int
    ratio::Number
end

function FixedSplit(data; ratio=0.8)
    0 < ratio < 1 || throw(ArgumentError("ratio should be between zero and one"))
    return FixedSplit(data, _nobs(data), ratio)
end

Base.length(r::FixedSplit) = 1
Base.eltype(r::FixedSplit{D}) where D = Tuple{D, D}

@propagate_inbounds function Base.iterate(r::FixedSplit, state=1)
    state > 1 && return nothing
    k = ceil(Int, r.ratio * r.nobs)
    train = _getobs(r.data, 1:k)
    test = _getobs(r.data, (k + 1):r.nobs)
    return ((train, test), state + 1)
end

struct RandomSplit{D} <: ResampleMethod
    data::D
    nobs::Int
    ratio::Number
    times::Int
end

function RandomSplit(data; ratio=0.8, times=1)
    0 < ratio < 1 || throw(ArgumentError("ratio should be between zero and one"))
    return RandomSplit(data, _nobs(data), ratio, times)
end

Base.length(r::RandomSplit) = r.times
Base.eltype(r::RandomSplit{D}) where D = Tuple{D, D}

@propagate_inbounds function Base.iterate(r::RandomSplit, state=1)
    state > r.times && return nothing
    indices = shuffle!([1:r.nobs;])
    k = ceil(Int, r.ratio * r.nobs)
    train = _getobs(r.data, indices[1:k])
    test = _getobs(r.data, indices[(k + 1):end])
    return ((train, test), state + 1)
end

struct LeavePOut{D} <: ResampleMethod
    data::D
    nobs::Int
    p::Int
    indices::Vector{Int}
    shuffle::Bool
end

function LeavePOut(data; p=1, shuffle=true)
    n = _nobs(data)
    0 < p < n || throw(ArgumentError("p should be between 0 and $n"))
    return LeavePOut(data, n, p, [1:n;], shuffle)
end

Base.length(r::LeavePOut) = floor(Int, r.nobs / r.p)
Base.eltype(r::LeavePOut{D}) where D = Tuple{D, D}

@propagate_inbounds function Base.iterate(r::LeavePOut, state=1)
    state > length(r) && return nothing
    if r.shuffle && state == 1
        shuffle!(r.indices)
    end
    fold = ((state - 1) * r.p + 1):(state * r.p)
    train = _getobs(r.data, r.indices[1:end .∉ Ref(fold)])
    test = _getobs(r.data, r.indices[fold])
    return ((train, test), state + 1)
end

struct KFold{D} <: ResampleMethod
    data::D
    nobs::Int
    k::Int
    indices::Vector{Int}
    shuffle::Bool
end

function KFold(data; k=10, shuffle=true)
    n = _nobs(data)
    1 < k < n + 1 || throw(ArgumentError("k should be between 1 and $(n + 1)"))
    return KFold(data, n, k, [1:n;], shuffle)
end

Base.length(r::KFold) = r.k
Base.eltype(r::KFold{D}) where D = Tuple{D, D}

@propagate_inbounds function Base.iterate(r::KFold)
    if r.shuffle
        shuffle!(r.indices)
    end
    p = floor(Int, r.nobs / r.k)
    if mod(r.nobs, r.k) ≥ 1
        p = p + 1
    end
    train = _getobs(r.data, r.indices[(p + 1):end])
    test = _getobs(r.data, r.indices[1:p])
    return ((train, test), (2, p))
end

@propagate_inbounds function Base.iterate(r::KFold, state)
    state[1] > r.k && return nothing
    p = floor(Int, r.nobs / r.k)
    if mod(r.nobs, r.k) ≥ state[1]
        p = p + 1
    end
    fold = (state[2] + 1):(state[2] + p)
    train = _getobs(r.data, r.indices[1:end .∉ Ref(fold)])
    test = _getobs(r.data, r.indices[fold])
    return ((train, test), (state[1] + 1, state[2] + p))
end

struct ExhaustiveSearch
    keys::Tuple
    iter::ProductIterator
end

function ExhaustiveSearch(; args...)
    return ExhaustiveSearch(keys(args), product(values(args)...))
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

_fit(data::AbstractArray, fit) = fit(data)
_fit(data::Union{Tuple, NamedTuple}, fit) = fit(data...)
_fit(data::AbstractArray, fit, args) = fit(data, args...)
_fit(data::Union{Tuple, NamedTuple}, fit, args) = fit(data..., args...)

_score(data::AbstractArray, model) = score(model, data)
_score(data::Union{Tuple, NamedTuple}, model) = score(model, data...)

struct ModelValidation{T1,T2}
    models::Array{T1, 1}
    scores::Array{T2, 1}
end

struct ParameterSearch{T1,T2}
    models::Array{T1, 2}
    scores::Array{T2, 2}
    final::T1
end

nopreprocess(train, test) = train, test

function crossvalidate(fit::Function, resample::ResampleMethod; preprocess=nopreprocess, verbose=false)
    n = length(resample)
    models = Array{Any, 1}(undef, n)
    scores = Array{Any, 1}(undef, n)

    i = 1
    for (train, test) in resample
        if (verbose) @info "Iteration $i of $n" end
        train, test = preprocess(train, test)
        models[i] = _fit(train, fit)
        scores[i] = _score(test, models[i])
        if i == 1
            models = convert(Array{typeof(models[1])}, models)
            scores = convert(Array{typeof(scores[1])}, scores)
        end
        i = i + 1
    end

    return ModelValidation(models, scores)
end

function crossvalidate(fit::Function, resample::ResampleMethod, search::ExhaustiveSearch; preprocess=nopreprocess, minimize=true, verbose=false)
    grid = collect(search)
    n, m = length(resample), length(grid)
    models = Array{Any, 2}(undef, n, m)
    scores = Array{Any, 2}(undef, n, m)

    i = 1
    for (train, test) in resample
        if (verbose) @info "Iteration $i of $n" end
        train, test = preprocess(train, test)
        models[i,:] = pmap((args) -> _fit(train, fit, args), grid)
        scores[i,:] = map((model) -> _score(test, model), models[i,:])
        if i == 1
            models = convert(Array{typeof(models[1])}, models)
            scores = convert(Array{typeof(scores[1])}, scores)
        end
        i = i + 1
    end

    index = 1
    if minimize
        index = argmin(sum(scores, dims=1) ./ n)[2]
    else
        index = argmax(sum(scores, dims=1) ./ n)[2]
    end

    if (verbose) @info "Fitting final model" end
    final = _fit(resample.data, fit, grid[index])

    return ParameterSearch(models, scores, final)
end

function predict(cv::ParameterSearch, kwargs...)
    return predict(cv.final, kwargs...)
end

function score(cv::ParameterSearch, kwargs...)
    return score(cv.final, kwargs...)
end

end
