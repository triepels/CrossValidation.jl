module CrossValidation

using Base: @propagate_inbounds
using Random: shuffle!
using Distributed: pmap

export Resampler, FixedSplit, RandomSplit, KFold, ForwardChaining, SlidingWindow, PreProcess,
       SearchSpace, Optimizer, ExhaustiveSearch, RandomSearch,
       loss, cv, optimize

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

abstract type Resampler end

Base.eltype(r::Resampler) = Tuple{restype(r.data), restype(r.data)}

struct FixedSplit{D} <: Resampler
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

data(r::FixedSplit) = r.data
Base.length(r::FixedSplit) = 1

@propagate_inbounds function Base.iterate(r::FixedSplit, state = 1)
    state > 1 && return nothing
    m = ceil(Int, r.ratio * r.nobs)
    train = getobs(r.data, 1:m)
    test = getobs(r.data, (m + 1):r.nobs)
    return (train, test), state + 1
end

struct RandomSplit{D} <: Resampler
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

data(r::RandomSplit) = r.data
Base.length(r::RandomSplit) = 1

@propagate_inbounds function Base.iterate(r::RandomSplit, state = 1)
    state > 1 && return nothing
    m = ceil(Int, r.ratio * r.nobs)
    train = getobs(r.data, r.perm[1:m])
    test = getobs(r.data, r.perm[(m + 1):r.nobs])
    return (train, test), state + 1
end

struct KFold{D} <: Resampler
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

data(r::KFold) = r.data
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

struct ForwardChaining{D} <: Resampler
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

data(r::ForwardChaining) = r.data

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

struct SlidingWindow{D} <: Resampler
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

data(r::SlidingWindow) = r.data

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

struct PreProcess <: Resampler
    res::Resampler
    f::Function
end

Base.eltype(p::PreProcess) = eltype(p.res)
Base.length(p::PreProcess) = length(p.res)

data(p::PreProcess) = p.f(data(p.res))

function Base.iterate(r::PreProcess, state = 1)
    next = iterate(r.res, state)
    next === nothing && return nothing
    (train, test), state = next
    return r.f(train, test), state
end

struct SearchSpace{names, T<:Tuple}
    args::T
end

function SearchSpace{names}(args::Vararg) where names
    if length(args) != length(names::Tuple)
        throw(ArgumentError("argument names and values must have matching lengths"))
    end
    return SearchSpace{names, typeof(args)}(args)
end

Base.eltype(::Type{SearchSpace{names, T}}) where {names, T} = NamedTuple{names, Tuple{map(eltype, T.parameters)...}}
Base.length(s::SearchSpace) = prod(length, s.args)

Base.firstindex(s::SearchSpace) = 1
Base.lastindex(s::SearchSpace) = length(s)

Base.size(s::SearchSpace) = map(length, s.args)

function Base.size(s::SearchSpace, d::Integer)
    @boundscheck d < 1 && throw(DimensionMismatch("dimension out of range"))
    return d > length(s.args) ? 1 : length(s.args[d])
end

@inline function Base.getindex(s::SearchSpace{names, T}, i::Int) where {names, T}
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    strides = (1, cumprod(map(length, Base.front(s.args)))...)
    return NamedTuple{names}(map(getindex, s.args, mod.((i - 1) .÷ strides, size(s)) .+ 1))
end

@inline function Base.getindex(s::SearchSpace{names, T}, I::Vararg{Int, N}) where {names, T, N}
    @boundscheck length(I) == length(s.args) && all(1 .≤ I .≤ size(s)) || throw(BoundsError(s, I))
    return NamedTuple{names}(map(getindex, s.args, I))
end

abstract type Optimizer end

@propagate_inbounds function Base.iterate(s::Optimizer, state = 1)
    state > length(s) && return nothing
    return s[state], state + 1
end

_fit(f, x::AbstractArray, args) = f(x; args...)
_fit(f, x::Union{Tuple, NamedTuple}, args) = f(x...; args...)

_loss(model, x::AbstractArray) = loss(model, x)
_loss(model, x::Union{Tuple, NamedTuple}) = loss(model, x...)

loss(model, x) = throw(ErrorException("no loss function defined for $(typeof(model))"))

function _eval(f, opt, train, test)
    models = pmap(args -> _fit(f, train, args), opt)
    return map(model -> _loss(model, test), models)
end

mean(f, itr) = sum(f, itr) / length(itr)

function eval(f::Function, opt::Optimizer, res::Resampler)
    return mean(x -> _eval(f, opt, x...), res)
end

struct ExhaustiveSearch <: Optimizer
    space::SearchSpace
end

Base.eltype(s::ExhaustiveSearch) = eltype(s.space)
Base.length(s::ExhaustiveSearch) = length(s.space)

@inline function Base.getindex(s::ExhaustiveSearch, i::Int)
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    return @inbounds s.space[i]
end

struct RandomSearch <: Optimizer
    space::SearchSpace
    cand::Vector{Int}
end

function RandomSearch(space::SearchSpace, n::Int = 1) where T
    m = length(space)
    1 ≤ n ≤ m || throw(ArgumentError("cannot sample $n times without replacement from search space"))
    cand = sizehint!(Int[], n)
    for _ in 1:n
        c = rand(1:m)
        while c in cand
            c = rand(1:m)
        end
        push!(cand, c)
    end
    return RandomSearch(space, cand)
end

Base.eltype(s::RandomSearch) = eltype(s.space)
Base.length(s::RandomSearch) = length(s.cand)

@inline function Base.getindex(s::RandomSearch, i::Int)
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    return @inbounds s.space[s.cand[i]]
end

_fit(f, x::AbstractArray) = f(x)
_fit(f, x::Union{Tuple, NamedTuple}) = f(x...)

function cv(f::Function, res::Resampler)
    return map(data -> _loss(_fit(f, data[1]), data[2]), res)
end

function optimize(f::Function, opt::Optimizer, res::Resampler; maximize::Bool = true)
    length(opt) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    best = maximize ? argmax(eval(f, opt, res)) : argmin(eval(f, opt, res))
    return _fit(f, data(res), opt[best])
end

end
