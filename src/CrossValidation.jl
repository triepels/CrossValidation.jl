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

Base.eltype(r::ResampleMethod) = Tuple{restype(r.x), restype(r.x)}

struct FixedSplit{D} <: ResampleMethod
    x::D
    n::Int
    ratio::Number
end

function FixedSplit(x::Union{AbstractArray, Tuple, NamedTuple}, ratio::Number = 0.8)
    n = nobs(x)
    1 ≤ n * ratio ≤ n - 1 || throw(ArgumentError("data cannot be split based on a $ratio ratio"))
    return FixedSplit(x, n, ratio)
end

function FixedSplit(x::Union{AbstractArray, Tuple, NamedTuple}, m::Int)
    n = nobs(x)
    1 ≤ m ≤ n - 1 || throw(ArgumentError("data cannot be split by $m"))
    return FixedSplit(x, n, m / n)
end

Base.length(r::FixedSplit) = 1

@propagate_inbounds function Base.iterate(r::FixedSplit, state = 1)
    state > 1 && return nothing
    m = ceil(Int, r.ratio * r.n)
    train = getobs(r.x, 1:m)
    test = getobs(r.x, (m + 1):r.n)
    return (train, test), state + 1
end

struct RandomSplit{D} <: ResampleMethod
    x::D
    n::Int
    ratio::Number
    times::Int
end

function RandomSplit(x::Union{AbstractArray, Tuple, NamedTuple}, ratio::Number = 0.8; times::Int = 1)
    n = nobs(x)
    1 ≤ n * ratio ≤ n - 1 || throw(ArgumentError("data cannot be split based on a $ratio ratio"))
    0 < times || throw(ArgumentError("unable to repeat resampling $times times"))
    return RandomSplit(x, n, ratio, times)
end

function RandomSplit(x::Union{AbstractArray, Tuple, NamedTuple}, m::Int; times::Int = 1)
    n = nobs(x)
    1 ≤ m ≤ n - 1 || throw(ArgumentError("data cannot be split by $m"))
    0 < times || throw(ArgumentError("unable to repeat resampling $times times"))
    return RandomSplit(x, n, m / n, times)
end

Base.length(r::RandomSplit) = r.times

@propagate_inbounds function Base.iterate(r::RandomSplit, state = 1)
    state > r.times && return nothing
    inds = shuffle!([1:r.n; ])
    m = ceil(Int, r.ratio * r.n)
    train = getobs(r.x, inds[1:m])
    test = getobs(r.x, inds[(m + 1):r.n])
    return (train, test), state + 1
end

struct StratifiedSplit{D} <: ResampleMethod
    x::D
    n::Int
    ratio::Number
    times::Int
    strata::Vector{Vector{Int}}
end

function StratifiedSplit(x::Union{AbstractArray, Tuple, NamedTuple}, y::AbstractVector, ratio::Number = 0.8; times::Int = 1)
    n = nobs(x)
    nobs(y) == n || throw(ArgumentError("x and y should have the same number of observations"))
    
    strata = map(s -> findall(y .== s), unique(y))
    for s in strata
        l = length(s)
        1 ≤ l * ratio ≤ l - 1 || throw(ArgumentError("unable to stratify data based on a $ratio ratio"))
    end

    0 < times || throw(ArgumentError("unable to repeat resampling $times times"))

    return StratifiedSplit(x, n, ratio, times, strata)
end

function StratifiedSplit(x::Union{AbstractArray, Tuple, NamedTuple}, y::AbstractVector, m::Int; times::Int = 1)
    n = nobs(x)
    nobs(y) == n || throw(ArgumentError("x and y should have the same number of observations"))
    
    strata = map(s -> findall(y .== s), unique(y))
    for s in strata
        l = length(s)
        1 ≤ l / n * m ≤ l - 1 || throw(ArgumentError("unable to stratify data by $m"))
    end

    0 < times || throw(ArgumentError("unable to repeat resampling $times times"))

    return StratifiedSplit(x, n, m / n, times, strata)
end

Base.length(r::StratifiedSplit) = r.times

@propagate_inbounds function Base.iterate(r::StratifiedSplit, state = 1)
    state > r.times && return nothing
    inds = sizehint!(Int[], ceil(Int, r.ratio * r.n))
    for s in r.strata
        shuffle!(s)
        m = ceil(Int, length(s) * r.ratio)
        append!(inds, s[1:m])
    end
    shuffle!(inds)
    train = getobs(r.x, inds)
    test = getobs(r.x, shuffle!(setdiff(1:r.n, inds)))
    return (train, test), state + 1
end

struct KFold{D} <: ResampleMethod
    x::D
    n::Int
    k::Int
    inds::Vector{Int}
end

function KFold(x::Union{AbstractArray, Tuple, NamedTuple}; k::Int = 10)
    n = nobs(x)
    1 < k ≤ n || throw(ArgumentError("data cannot be partitioned into $k folds"))
    return KFold(x, n, k, [1:n;])
end

Base.length(r::KFold) = r.k

@propagate_inbounds function Base.iterate(r::KFold, state = 1)
    state > r.k && return nothing
    if state == 1
        shuffle!(r.inds)
    end
    m = mod(r.n, r.k)
    w = floor(Int, r.n / r.k)
    fold = ((state - 1) * w + min(m, state - 1) + 1):(state * w + min(m, state))
    train = getobs(r.x, r.inds[1:end .∉ Ref(fold)])
    test = getobs(r.x, r.inds[fold])
    return (train, test), state + 1
end

struct StratifiedKFold{D} <: ResampleMethod
    x::D
    n::Int
    k::Int
    strata::Vector{Vector{Int}}
end

function StratifiedKFold(x::Union{AbstractArray, Tuple, NamedTuple}, y::AbstractVector; k::Int = 10)
    n = nobs(x)
    nobs(y) == n || throw(ArgumentError("data should have the same number of observations"))

    strata = map(s -> findall(y .== s), unique(y))
    for s in strata
        length(s) ≥ k || throw(ArgumentError("not all strata can be partitioned into $k folds"))
    end

    return StratifiedKFold(x, n, k, strata)
end

Base.length(r::StratifiedKFold) = r.k

@propagate_inbounds function Base.iterate(r::StratifiedKFold, state = 1)
    state > r.k && return nothing
    inds = sizehint!(Int[], ceil(Int, (1 / r.k) * r.n))
    for s in r.strata
        if state == 1
            shuffle!(s)
        end
        m = mod(length(s), r.k)
        w = floor(Int, length(s) / r.k)
        fold = ((state - 1) * w + min(m, state - 1) + 1):(state * w + min(m, state))
        append!(inds, s[fold])
    end
    shuffle!(inds)
    train = getobs(r.x, shuffle!(setdiff(1:r.n, inds)))
    test = getobs(r.x, inds)
    return (train, test), state + 1
end

struct ForwardChaining{D} <: ResampleMethod
    x::D
    n::Int
    init::Int
    out::Int
    partial::Bool
end

function ForwardChaining(x::Union{AbstractArray, Tuple, NamedTuple}, init::Int, out::Int; partial::Bool = true)
    n = nobs(x)
    1 ≤ init ≤ n || throw(ArgumentError("invalid initial window of $init"))
    1 ≤ out ≤ n || throw(ArgumentError("invalid out-of-sample window of $out"))
    init + out ≤ n || throw(ArgumentError("initial and out-of-sample window exceed the number of data observations"))
    return ForwardChaining(x, n, init, out, partial)
end

function Base.length(r::ForwardChaining)
    l = (r.n - r.init) / r.out
    if r.partial
        ceil(Int, l)
    else
        floor(Int, l)
    end
end

@propagate_inbounds function Base.iterate(r::ForwardChaining, state = 1)
    state > length(r) && return nothing
    train = getobs(r.x, 1:(r.init + (state - 1) * r.out))
    test = getobs(r.x, (r.init + (state - 1) * r.out + 1):min(r.init + state * r.out, r.n))
    return (train, test), state + 1
end

struct SlidingWindow{D} <: ResampleMethod
    x::D
    n::Int
    window::Int
    out::Int
    partial::Bool
end

function SlidingWindow(x::Union{AbstractArray, Tuple, NamedTuple}, window::Int, out::Int; partial::Bool = true)
    n = nobs(x)
    1 ≤ window ≤ n || throw(ArgumentError("invalid sliding window of $window"))
    1 ≤ out ≤ n || throw(ArgumentError("invalid out-of-sample window of $out"))
    window + out ≤ n || throw(ArgumentError("sliding and out-of-sample window exceed the number of data observations"))
    return SlidingWindow(x, n, window, out, partial)
end

function Base.length(r::SlidingWindow)
    l = (r.n - r.window) / r.out
    if r.partial
        ceil(Int, l)
    else
        floor(Int, l)
    end
end

@propagate_inbounds function Base.iterate(r::SlidingWindow, state = 1)
    state > length(r) && return nothing
    train = getobs(r.x, (1 + (state - 1) * r.out):(r.window + (state - 1) * r.out))
    test = getobs(r.x, (r.window + (state - 1) * r.out + 1):min(r.window + state * r.out, r.n))
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

    final = _fit(preprocess(resample.x), fit, grid[idx])

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
