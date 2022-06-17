module CrossValidation

using Base: @propagate_inbounds, Iterators.ProductIterator
using Random: shuffle!
using Distributed: pmap

export ResampleMethod, FixedSplit, RandomSplit, KFold,
       ExhaustiveSearch,
       ModelValidation, ParameterTuning, crossvalidate,
       predict, score

nobs(x::AbstractArray) = size(x)[end]

function nobs(x::Union{Tuple, NamedTuple})
    equal_nobs(x) || throw(ArgumentError("all data should have the same number of observations"))
    return nobs(x[1])
end

function equal_nobs(x::Union{Tuple, NamedTuple})
    length(x) > 0 || return false
    n = nobs(x[1])
    return all(y -> nobs(y) == n, Base.tail(x))
end

slice_obs(x::AbstractArray, i) = x[Base.setindex(map(Base.Slice, axes(x)), i, ndims(x))...]
slice_obs(x::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(slice_obs, i), x)

abstract type ResampleMethod end

struct FixedSplit{D} <: ResampleMethod
    data::D
    nobs::Int
    ratio::Number
end

function FixedSplit(data::Union{AbstractArray, Tuple, NamedTuple}; ratio::Number = 0.8)
    0 < ratio < 1 || throw(ArgumentError("ratio should be in (0, 1)"))
    return FixedSplit(data, nobs(data), ratio)
end

Base.length(r::FixedSplit) = 1
Base.eltype(r::FixedSplit{D}) where D = Tuple{D, D}

@propagate_inbounds function Base.iterate(r::FixedSplit, state=1)
    state > 1 && return nothing
    k = ceil(Int, r.ratio * r.nobs)
    train = slice_obs(r.data, 1:k)
    test = slice_obs(r.data, (k + 1):r.nobs)
    return ((train, test), state + 1)
end

struct RandomSplit{D} <: ResampleMethod
    data::D
    nobs::Int
    ratio::Number
    times::Int
end

function RandomSplit(data::Union{AbstractArray, Tuple, NamedTuple}; ratio::Number = 0.8, times::Int = 1)
    0 < ratio < 1 || throw(ArgumentError("ratio should be in (0, 1)"))
    0 ≤ times || throw(ArgumentError("times should be non-negative"))
    return RandomSplit(data, nobs(data), ratio, times)
end

Base.length(r::RandomSplit) = r.times
Base.eltype(r::RandomSplit{D}) where D = Tuple{D, D}

@propagate_inbounds function Base.iterate(r::RandomSplit, state=1)
    state > r.times && return nothing
    k = ceil(Int, r.ratio * r.nobs)
    indices = shuffle!([1:r.nobs; ])
    train = slice_obs(r.data, indices[1:k])
    test = slice_obs(r.data, indices[(k + 1):end])
    return ((train, test), state + 1)
end

struct KFold{D} <: ResampleMethod
    data::D
    nobs::Int
    k::Int
    indices::Vector{Int}
    shuffle::Bool
end

function KFold(data::Union{AbstractArray, Tuple, NamedTuple}; k::Int = 10, shuffle::Bool = true)
    n = nobs(data)
    1 < k ≤ n || throw(ArgumentError("k should be in (1, $n]"))
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
    train = slice_obs(r.data, r.indices[(p + 1):end])
    test = slice_obs(r.data, r.indices[1:p])
    return ((train, test), (2, p))
end

@propagate_inbounds function Base.iterate(r::KFold, state)
    state[1] > r.k && return nothing
    p = floor(Int, r.nobs / r.k)
    if mod(r.nobs, r.k) ≥ state[1]
        p = p + 1
    end
    fold = (state[2] + 1):(state[2] + p)
    train = slice_obs(r.data, r.indices[1:end .∉ Ref(fold)])
    test = slice_obs(r.data, r.indices[fold])
    return ((train, test), (state[1] + 1, state[2] + p))
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
    model::Array{T1, 1}
    score::Array{T2, 1}
end

struct ParameterSearch{T1,T2}
    model::Array{T1, 2}
    score::Array{T2, 2}
    final::T1
end

nopreprocess(train) = train
nopreprocess(train, test) = train, test

function crossvalidate(fit::Function, resample::ResampleMethod; preprocess::Function = nopreprocess, verbose::Bool = false)
    n = length(resample)
    model = Array{Any, 1}(undef, n)
    score = Array{Any, 1}(undef, n)

    i = 1
    for (train, test) in resample
        train, test = preprocess(train, test)

        model[i] = _fit(train, fit)
        score[i] = _score(test, model[i])

        if i == 1
            model = convert(Array{typeof(model[1])}, model)
            score = convert(Array{typeof(score[1])}, score)
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
    model = Array{Any, 2}(undef, n, m)
    score = Array{Any, 2}(undef, n, m)

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
