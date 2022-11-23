module CrossValidation

using Base: @propagate_inbounds
using Random: shuffle!
using Distributed: @distributed, pmap

export DataSampler, FixedSplit, RandomSplit, KFold, ForwardChaining, SlidingWindow, PreProcess,
       ParameterSpace, ParameterSampler, GridSampler, RandomSampler,
       fit!, loss, cv, brute, hc, Budget, sha

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

abstract type DataSampler end

Base.eltype(r::DataSampler) = Tuple{restype(r.data), restype(r.data)}

struct FixedSplit{D} <: DataSampler
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

getdata(r::FixedSplit) = r.data
Base.length(r::FixedSplit) = 1

@propagate_inbounds function Base.iterate(r::FixedSplit, state = 1)
    state > 1 && return nothing
    m = ceil(Int, r.ratio * r.nobs)
    train = getobs(r.data, 1:m)
    test = getobs(r.data, (m + 1):r.nobs)
    return (train, test), state + 1
end

struct RandomSplit{D} <: DataSampler
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

getdata(r::RandomSplit) = r.data
Base.length(r::RandomSplit) = 1

@propagate_inbounds function Base.iterate(r::RandomSplit, state = 1)
    state > 1 && return nothing
    m = ceil(Int, r.ratio * r.nobs)
    train = getobs(r.data, r.perm[1:m])
    test = getobs(r.data, r.perm[(m + 1):r.nobs])
    return (train, test), state + 1
end

struct KFold{D} <: DataSampler
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

getdata(r::KFold) = r.data
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

struct ForwardChaining{D} <: DataSampler
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

getdata(r::ForwardChaining) = r.data

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

struct SlidingWindow{D} <: DataSampler
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

getdata(r::SlidingWindow) = r.data

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

struct PreProcess <: DataSampler
    res::DataSampler
    f::Function
end

Base.eltype(p::PreProcess) = eltype(p.res)
Base.length(p::PreProcess) = length(p.res)

getdata(p::PreProcess) = p.f(getdata(p.res))

function Base.iterate(r::PreProcess, state = 1)
    next = iterate(r.res, state)
    next === nothing && return nothing
    (train, test), state = next
    return r.f(train, test), state
end

struct ParameterSpace{names, T<:Tuple}
    args::T
end

function ParameterSpace{names}(args::Vararg) where names
    if length(args) != length(names::Tuple)
        throw(ArgumentError("argument names and values must have matching lengths"))
    end
    return ParameterSpace{names, typeof(args)}(args)
end

Base.eltype(::Type{ParameterSpace{names, T}}) where {names, T} = NamedTuple{names, Tuple{map(eltype, T.parameters)...}}
Base.length(s::ParameterSpace) = prod(length, s.args)

Base.firstindex(s::ParameterSpace) = 1
Base.lastindex(s::ParameterSpace) = length(s)

Base.size(s::ParameterSpace) = map(length, s.args)

function Base.size(s::ParameterSpace, d::Integer)
    @boundscheck d < 1 && throw(DimensionMismatch("dimension out of range"))
    return d > length(s.args) ? 1 : length(s.args[d])
end

@inline function Base.getindex(s::ParameterSpace{names, T}, i::Int) where {names, T}
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    strides = (1, cumprod(map(length, Base.front(s.args)))...)
    return NamedTuple{names}(map(getindex, s.args, mod.((i - 1) .÷ strides, size(s)) .+ 1))
end

@inline function Base.getindex(s::ParameterSpace{names, T}, I::Vararg{Int, N}) where {names, T, N}
    @boundscheck length(I) == length(s.args) && all(1 .≤ I .≤ size(s)) || throw(BoundsError(s, I))
    return NamedTuple{names}(map(getindex, s.args, I))
end

@inline function Base.getindex(s::ParameterSpace{names, T}, inds::Vector{Int}) where {names, T}
    return [s[i] for i in inds]
end

abstract type ParameterSampler end

@propagate_inbounds function Base.iterate(s::ParameterSampler, state = 1)
    state > length(s) && return nothing
    return s[state], state + 1
end

# To do: add gap between elements?
struct GridSampler <: ParameterSampler
    space::ParameterSpace
end

Base.eltype(s::GridSampler) = eltype(s.space)
Base.length(s::GridSampler) = length(s.space)

@inline function Base.getindex(s::GridSampler, i::Int)
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    return @inbounds s.space[i]
end

struct RandomSampler <: ParameterSampler
    space::ParameterSpace
    inds::Vector{Int}
end

function RandomSampler(space::ParameterSpace; n::Int = 1)
    m = length(space)
    1 ≤ n ≤ m || throw(ArgumentError("cannot sample $n times without replacement from search space"))
    inds = sizehint!(Int[], n)
    for _ in 1:n
        i = rand(1:m)
        while i in inds
            i = rand(1:m)
        end
        push!(inds, i)
    end
    return RandomSampler(space, inds)
end

Base.eltype(s::RandomSampler) = eltype(s.space)
Base.length(s::RandomSampler) = length(s.inds)

@inline function Base.getindex(s::RandomSampler, i::Int)
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    return @inbounds s.space[s.inds[i]]
end

_fit(f, x::AbstractArray) = f(x)
_fit(f, x::Union{Tuple, NamedTuple}) = f(x...)
_fit(f, x::AbstractArray, args) = f(x; args...)
_fit(f, x::Union{Tuple, NamedTuple}, args) = f(x...; args...)

_loss(model, x::AbstractArray) = loss(model, x)
_loss(model, x::Union{Tuple, NamedTuple}) = loss(model, x...)

loss(model, x...) = throw(ErrorException("no loss function defined for $(typeof(model))"))

function _evalfold(f, parms, train, test)
    models = pmap(args -> _fit(f, train, args), parms)
    return map(model -> _loss(model, test), models)
end

mean(f, itr) = sum(f, itr) / length(itr)

function _eval(f, parms, data)
    loss = mean(fold -> _evalfold(f, parms, fold...), data)
    @debug "Evaluated models" parms=collect(parms) loss
    return loss
end

function cv(f::Function, data::DataSampler)
    return map(fold -> _loss(_fit(f, fold[1]), fold[2]), data)
end

function brute(f::Function, parms::ParameterSampler, data::DataSampler; maximize::Bool = true)
    length(parms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    @debug "Start brute-force search"
    best = maximize ? argmax(_eval(f, parms, data)) : argmin(_eval(f, parms, data))
    @debug "Finished brute-force search"
    return _fit(f, getdata(data), parms[best])
end

function _candidates(space, i)
    dim = size(space)
    cand = sizehint!(Int[], 2 * length(dim))
    @inbounds for j in eachindex(dim)
        if j == 1
            ind = mod(i - 1, dim[1]) + 1
            if ind > 1
                push!(cand, i - 1)
            end
            if ind < dim[1]
                push!(cand, i + 1)
            end
        else
            ind = mod((i - 1) ÷ dim[j - 1], dim[j]) + 1
            if ind > 1
                push!(cand, i - dim[j - 1])
            end
            if ind < dim[j]
                push!(cand, i + dim[j - 1])
            end
        end
    end
    return cand
end

function hc(f::Function, space::ParameterSpace, data::DataSampler; maximize::Bool = true)
    n = length(space)
    n ≥ 1 || throw(ArgumentError("nothing to optimize"))

    best = rand(1:n)
    loss = maximize ? -Inf : Inf
    cand = _candidates(space, best)

    @debug "Start hill-climbing"
    while !isempty(cand)
        clss = _eval(f, space[cand], data)

        if maximize
            cbst = argmax(clss)
            if loss ≥ clss[cbst] 
                break
            end
        else
            cbst = argmin(clss)
            if loss ≤ clss[cbst]
                break
            end
        end
        
        best = cand[cbst]
        loss = clss[cbst]

        cand = _candidates(space, best)
    end
    @debug "Finished hill-climbing"

    return _fit(f, getdata(data), space[best])
end

_fit!(model, x::AbstractArray, args) = fit!(model, x; args...)
_fit!(model, x::Union{Tuple, NamedTuple}, args) = fit!(model, x...; args...)

fit!(model, x; args...) = throw(ErrorException("no fit! function defined for $(typeof(model))"))

const Budget = NamedTuple{names, T} where {names, T<:Tuple{Vararg{Int}}}

struct Arm{M,P}
    model::M
    parms::P
end

function _evalarms(arms, args, data)
    train, test = first(data)
    @distributed for arm in arms
        _fit!(arm.model, train, args)
    end
    loss = map(arm -> _loss(arm.model, test), arms)
    @debug "Evaluated arms" arms args loss
    return loss
end

_halve!(x) = resize!(x, ceil(Int, length(x) / 2))

function sha(M::Type, parms::ParameterSampler, budget::Budget, data::DataSampler; maximize::Bool = true)
    n = length(parms)
    n ≥ 1 || throw(ArgumentError("nothing to optimize"))

    m = ceil(Int, log2(n))

    arms = map(x -> Arm(M(x...), x), parms)
    args = map(x -> floor(Int, x / m), budget)

    @debug "Start successive halving"   
    for _ in 1:m
        loss = _evalarms(arms, args, data)
        arms = _halve!(arms[sortperm(loss, rev=maximize)])
    end
    @debug "Finished successive halving"

    return _fit!(M(arms[1].parms...), getdata(data), budget)
end

end
