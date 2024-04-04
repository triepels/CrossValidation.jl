module CrossValidation

using Base: @propagate_inbounds, OneTo
using Random: GLOBAL_RNG, AbstractRNG, SamplerTrivial, shuffle!
using Distributed: pmap

import Random: rand

export AbstractResampler, MonadicResampler, VariadicResampler, FixedSplit, RandomSplit, LeaveOneOut, KFold, ForwardChaining, SlidingWindow,
       AbstractSpace, FiniteSpace, InfiniteSpace, space,
       AbstractDistribution, DiscreteDistribution, ContinousDistribution, Discrete, DiscreteUniform, Uniform, LogUniform, Normal,
       Budget, AbstractAllocation, GeometricAllocation, ContstantAllocation, HyperbandAllocation, allocate,
       fit!, loss, validate, brute, brutefit, hc, hcfit, sha, shafit, hyperband, hyperbandfit, sasha, sashafit

nobs(x::AbstractArray) = size(x)[end]
nobs(x) = length(x)

function nobs(x::Union{Tuple, NamedTuple})
    length(x) > 0 || return 0
    n = nobs(first(x))
    if !all(y -> nobs(y) == n, Base.tail(x))
        throw(ArgumentError("all data should have the same number of observations"))
    end
    return n
end

getobs(x::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(getobs, i), x)
getobs(x, i) = x[Base.setindex(ntuple(x -> Colon(), ndims(x)), i, ndims(x))...]

# TODO: is there a better solution to infer resample type?
restype(x) = restype(typeof(x))
restype(x::Type{T}) where T<:AbstractArray{CartesianIndex{N}, N} where N = Matrix{CartesianIndex{N}}
restype(x::Type{T}) where T<:AbstractRange = Vector{eltype(x)}
restype(x::Type{T}) where T<:LinearIndices{N} where N = Array{Int64, N}
restype(x::Type{T}) where T<:PermutedDimsArray{V, N} where {V, N} = Array{V, N}
restype(x::Type{T}) where T<:SubArray{V, N} where {V, N} = Array{V, N}
restype(x::Type{T}) where T = T

abstract type AbstractResampler{T} end

Base.eltype(::Type{R}) where R<:AbstractResampler{T} where T = Tuple{restype(T), restype(T)}

abstract type MonadicResampler{T} <: AbstractResampler{T} end
abstract type VariadicResampler{T} <: AbstractResampler{T} end

struct FixedSplit{T} <: MonadicResampler{T}
    data::T
    m::Int
    function FixedSplit(data, m::Int)
        n = nobs(data)
        1 ≤ m < n || throw(ArgumentError("data cannot be split by $m"))
        return new{typeof(data)}(data, m)
    end
end

FixedSplit(data, ratio::Real) = FixedSplit(data, floor(Int, nobs(data) * ratio))

struct RandomSplit{T} <: MonadicResampler{T}
    data::T
    m::Int
    perm::Vector{Int}
    function RandomSplit(data, m::Int)
        n = nobs(data)
        1 ≤ m < n || throw(ArgumentError("data cannot be split by $m"))
        return new{typeof(data)}(data, m, shuffle!([OneTo(n);]))
    end
end

RandomSplit(data, ratio::Real) = RandomSplit(data, floor(Int, nobs(data) * ratio))

struct LeaveOneOut{T} <: VariadicResampler{T}
    data::T
    function LeaveOneOut(data)
        n = nobs(data)
        n > 1 || throw(ArgumentError("data has too few observations to split"))
        return new{typeof(data)}(data)
    end
end

struct KFold{T} <: VariadicResampler{T}
    data::T
    k::Int
    perm::Vector{Int}
    function KFold(data, k::Int)
        n = nobs(data)
        1 < k ≤ n || throw(ArgumentError("data cannot be partitioned into $k folds"))
        return new{typeof(data)}(data, k, shuffle!([OneTo(n);]))
    end
end

struct ForwardChaining{T} <: VariadicResampler{T}
    data::T
    init::Int
    out::Int
    partial::Bool
    function ForwardChaining(data, init::Int, out::Int; partial::Bool = true)
        n = nobs(data)
        1 ≤ init ≤ n || throw(ArgumentError("invalid initial window of $init"))
        1 ≤ out ≤ n || throw(ArgumentError("invalid out-of-sample window of $out"))
        init + out ≤ n || throw(ArgumentError("initial and out-of-sample window exceed number of observations"))
        return new{typeof(data)}(data, init, out, partial)
    end
end

struct SlidingWindow{T} <: VariadicResampler{T}
    data::T
    window::Int
    out::Int
    partial::Bool
    function SlidingWindow(data, window::Int, out::Int; partial::Bool = true)
        n = nobs(data)
        1 ≤ window ≤ n || throw(ArgumentError("invalid sliding window of $window"))
        1 ≤ out ≤ n || throw(ArgumentError("invalid out-of-sample window of $out"))
        window + out ≤ n || throw(ArgumentError("sliding and out-of-sample window exceed number of observations"))
        return new{typeof(data)}(data, window, out, partial)
    end
end

Base.length(r::FixedSplit) = 1
Base.length(r::RandomSplit) = 1
Base.length(r::LeaveOneOut) = nobs(r.data)
Base.length(r::KFold) = r.k

function Base.length(r::ForwardChaining)
    l = (nobs(r.data) - r.init) / r.out
    return r.partial ? ceil(Int, l) : floor(Int, l)
end

function Base.length(r::SlidingWindow)
    l = (nobs(r.data) - r.window) / r.out
    return r.partial ? ceil(Int, l) : floor(Int, l)
end

@propagate_inbounds function Base.iterate(r::FixedSplit, state = 1)
    state > 1 && return nothing
    x = getobs(r.data, OneTo(r.m))
    y = getobs(r.data, (r.m + 1):nobs(r.data))
    return (x, y), state + 1
end

@propagate_inbounds function Base.iterate(r::RandomSplit, state = 1)
    state > 1 && return nothing
    x = getobs(r.data, r.perm[OneTo(r.m)])
    y = getobs(r.data, r.perm[(r.m + 1):nobs(r.data)])
    return (x, y), state + 1
end

@propagate_inbounds function Base.iterate(r::LeaveOneOut, state = 1)
    state > length(r) && return nothing
    x = getobs(r.data, union(OneTo(state - 1), (state + 1):nobs(r.data)))
    y = getobs(r.data, state:state)
    return (x, y), state + 1
end

@propagate_inbounds function Base.iterate(r::KFold, state = 1)
    state > length(r) && return nothing
    n = nobs(r.data)
    m, w = mod(n, r.k), floor(Int, n / r.k)
    fold = ((state - 1) * w + min(m, state - 1) + 1):(state * w + min(m, state))
    x = getobs(r.data, r.perm[setdiff(OneTo(n), fold)])
    y = getobs(r.data, r.perm[fold])
    return (x, y), state + 1
end

@propagate_inbounds function Base.iterate(r::ForwardChaining, state = 1)
    state > length(r) && return nothing
    x = getobs(r.data, OneTo(r.init + (state - 1) * r.out))
    y = getobs(r.data, (r.init + (state - 1) * r.out + 1):min(r.init + state * r.out, nobs(r.data)))
    return (x, y), state + 1
end

@propagate_inbounds function Base.iterate(r::SlidingWindow, state = 1)
    state > length(r) && return nothing
    x = getobs(r.data, (1 + (state - 1) * r.out):(r.window + (state - 1) * r.out))
    y = getobs(r.data, (r.window + (state - 1) * r.out + 1):min(r.window + state * r.out, nobs(r.data)))
    return (x, y), state + 1
end

abstract type AbstractDistribution{T} end

Base.eltype(::Type{D}) where D<:AbstractDistribution{T} where T = T

abstract type DiscreteDistribution{T} <: AbstractDistribution{T} end
abstract type ContinousDistribution{T<:AbstractFloat} <: AbstractDistribution{T} end

Base.getindex(d::DiscreteDistribution, i) = getindex(values(d), i)
Base.iterate(d::DiscreteDistribution) = iterate(values(d))
Base.iterate(d::DiscreteDistribution, state) = iterate(values(d), state)
Base.length(d::DiscreteDistribution) = length(values(d))

struct Discrete{T} <: DiscreteDistribution{T}
    vals::Vector{T}
    probs::Vector{Float64}
    function Discrete(vals::V, probs::Vector{P}) where {V, P<:Real}
        length(vals) ≥ 1 || throw(ArgumentError("no values provided"))
        length(vals) == length(probs) || throw(ArgumentError("lenghts of values and probabilities do not match"))
        (all(probs .≥ 0) && isapprox(sum(probs), 1)) || throw(ArgumentError("invalid probabilities provided"))
        return new{eltype(V)}(vals, probs)
    end
end

struct DiscreteUniform{T} <: DiscreteDistribution{T}
    vals::Vector{T}
    function DiscreteUniform(vals::V) where V
        length(vals) ≥ 1 || throw(ArgumentError("no values provided"))
        return new{eltype(V)}(vals)
    end
end

Base.values(d::Discrete) = d.vals
Base.values(d::DiscreteUniform) = d.vals

struct Uniform{T} <: ContinousDistribution{T}
    a::T
    b::T
    function Uniform(a::Real, b::Real)
        a, b = promote(float(a), float(b))
        a < b || throw(ArgumentError("a must be smaller than b"))
        return new{typeof(a)}(a, b)
    end
end

struct LogUniform{T} <: ContinousDistribution{T}
    a::T
    b::T
    function LogUniform(a::Real, b::Real)
        a, b = promote(float(a), float(b))
        a < b || throw(ArgumentError("a must be smaller than b"))
        return new{typeof(a)}(a, b)
    end
end

struct Normal{T} <: ContinousDistribution{T}
    mean::T
    std::T
    function Normal(mean::Real, std::Real)
        mean, std = promote(float(mean), float(std))
        std > zero(std) || throw(ArgumentError("standard deviation must be larger than zero"))
        return new{typeof(mean)}(mean, std)
    end
end

@inline function _rand_discrete(rng, vals, probs)
    c = 0.0
    q = rand(rng)
    for (val, p) in zip(vals, probs)
        c += p
        if q < c
            return val
        end
    end
    return last(vals)
end

rand(rng::AbstractRNG, d::SamplerTrivial{Discrete{T}}) where {T} = _rand_discrete(rng, d[].vals, d[].probs)
rand(rng::AbstractRNG, d::SamplerTrivial{DiscreteUniform{T}}) where T = rand(rng, d[].vals)
rand(rng::AbstractRNG, d::SamplerTrivial{Uniform{T}}) where T = T(d[].a + (d[].b - d[].a) * rand(rng, T))
rand(rng::AbstractRNG, d::SamplerTrivial{LogUniform{T}}) where T = T(exp(log(d[].a) + (log(d[].b) - log(d[].a)) * rand(rng, T)))
rand(rng::AbstractRNG, d::SamplerTrivial{Normal{T}}) where T = T(d[].mean + d[].std * randn(rng, T))

abstract type AbstractSpace{names, T<:Tuple} end

Base.eltype(::Type{S}) where S<:AbstractSpace{names, T} where {names, T} = NamedTuple{names, Tuple{map(eltype, fieldtypes(T))...}}

struct FiniteSpace{names, T} <: AbstractSpace{names, T}
    vars::T
end

Base.firstindex(s::FiniteSpace) = 1
Base.keys(s::FiniteSpace) = OneTo(length(s))
Base.lastindex(s::FiniteSpace) = length(s)
Base.length(s::FiniteSpace) = length(s.vars) == 0 ? 0 : prod(length, s.vars)
Base.size(s::FiniteSpace) = length(s.vars) == 0 ? (0,) : map(length, s.vars)

@inline function Base.getindex(s::FiniteSpace{names}, i::Int) where names
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    strides = (1, cumprod(map(length, Base.front(s.vars)))...)
    return NamedTuple{names}(map(getindex, s.vars, mod.((i - 1) .÷ strides, size(s)) .+ 1))
end

@inline function Base.getindex(s::FiniteSpace{names}, I::Vararg{Int}) where names
    @boundscheck length(I) == length(s.vars) && all(1 .≤ I .≤ size(s)) || throw(BoundsError(s, I))
    return NamedTuple{names}(map(getindex, s.vars, I))
end

@inline function Base.getindex(s::FiniteSpace{names}, inds) where names
    return [s[i] for i in inds]
end

@propagate_inbounds function Base.iterate(s::FiniteSpace, state = 1)
    state > length(s) && return nothing
    return s[state], state + 1
end

struct InfiniteSpace{names, T} <: AbstractSpace{names, T}
    vars::T
end

rand(rng::AbstractRNG, s::SamplerTrivial{FiniteSpace{names, T}}) where {names, T} = NamedTuple{names}(map(x -> rand(rng, x), s[].vars))
rand(rng::AbstractRNG, s::SamplerTrivial{InfiniteSpace{names, T}}) where {names, T} = NamedTuple{names}(map(x -> rand(rng, x), s[].vars))

space(; vars...) = space(keys(vars), values(values(vars)))
space(names, vars::Tuple{Vararg{DiscreteDistribution}}) = FiniteSpace{names, typeof(vars)}(vars)
space(names, vars::Tuple{Vararg{AbstractDistribution}}) = InfiniteSpace{names, typeof(vars)}(vars)

_fit!(model, x::Union{Tuple, NamedTuple}, args) = fit!(model, x...; args...)
_fit!(model, x, args) = fit!(model, x; args...)

fit!(model, x) = throw(MethodError(fit!, (model, x)))

_loss(model, x::Union{Tuple, NamedTuple}) = loss(model, x...)
_loss(model, x) = loss(model, x)

loss(model, x) = throw(MethodError(loss, (model, x)))

@inline function _val(f, space, data, args)
    return sum(x -> _fit_split(f, space, x..., args)[2], data) / length(data)
end

@inline function _fit_split(f, space, train, test, args)
    models = pmap(x -> _fit!(x, train, args), map(f, space))
    loss = map(x -> _loss(x, test), models)
    @debug "Fitted models" space args loss
    return models, loss
end

function validate(f::Function, data::AbstractResampler)
    @debug "Start model validation"
    loss = map(x -> _loss(f(x[1]), x[2]), data)
    @debug "Finished model validation"
    return loss
end

function brute(f::Function, space::Union{FiniteSpace, Vector{T}}, data::AbstractResampler; args::NamedTuple = NamedTuple(), maximize::Bool = false) where T<:NamedTuple
    length(space) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    
    @debug "Start brute-force search"
    loss = _val(f, space, data, args)
    @debug "Finished brute-force search"
    
    return space[maximize ? argmax(loss) : argmin(loss)]
end

function brutefit(f::Function, space::Union{FiniteSpace, Vector{T}}, data::MonadicResampler; args::NamedTuple = NamedTuple(), maximize::Bool = false) where T<:NamedTuple
    length(space) ≥ 1 || throw(ArgumentError("nothing to optimize"))

    @debug "Start brute-force search"
    models, loss = _fit_split(f, space, first(data)..., args)
    @debug "Finished brute-force search"
    
    return models[maximize ? argmax(loss) : argmin(loss)]
end

@propagate_inbounds function neighbors(rng::AbstractRNG, space::AbstractSpace, at::Tuple, step::Tuple, n::Int)
    return [neighbors(rng, space, at, step) for _ in OneTo(n)]
end

@propagate_inbounds function neighbors(rng::AbstractRNG, space::AbstractSpace, at::Tuple, step::Tuple)
    return eltype(space)(map((d, a, s) -> neighbors(rng, d, a, s), space.vars, at, step))
end

# TODO: do something like @domaincheck?
@propagate_inbounds function neighbors(rng::AbstractRNG, d::Discrete, at, step::Int)
    ind = findfirst(==(at), d.vals)
    isnothing(ind) && throw(DomainError(at, "$d is undefined at $at"))
    a, b = max(1, ind - abs(step)), min(ind + abs(step), length(d))
    vals, probs = view(d.vals, a:b), view(d.probs, a:b)
    return _rand_discrete(rng, vals, probs / sum(probs))
end

# TODO: do something like @domaincheck?
@propagate_inbounds function neighbors(rng::AbstractRNG, d::DiscreteUniform, at, step::Int)
    ind = findfirst(==(at), d.vals)
    isnothing(ind) && throw(DomainError(at, "$d is undefined at $at"))
    a, b = max(1, ind - abs(step)), min(ind + abs(step), length(d))
    return d[rand(rng, a:b)]
end

# TODO: do something like @domaincheck?
@propagate_inbounds function neighbors(rng::AbstractRNG, d::Uniform{T}, at::Real, step::Real) where T<:Real
    d.a ≤ at ≤ d.b || throw(DomainError(at, "$d is undefined at $at"))
    a, b = max(d.a, at - abs(step)), min(at + abs(step), d.b)
    return T((b - a) * rand(rng, T) + a)
end

function hc(rng::AbstractRNG, f::Function, space::AbstractSpace, data::AbstractResampler, step::Tuple; args::NamedTuple = NamedTuple(), n::Int = 1, maximize::Bool = false)
    n ≥ 1 || throw(ArgumentError("invalid sample size of $n"))

    parm = rand(rng, space)
    best = maximize ? -Inf : Inf

    @debug "Start hill-climbing"
    nb = neighbors(rng, space, values(parm), step, n)
    @inbounds while !isempty(nb)
        loss = _val(f, nb, data, args)
        if maximize
            i = argmax(loss)
            loss[i] > best || break
        else
            i = argmin(loss)
            loss[i] < best || break
        end
        parm, best = nb[i], loss[i]
        nb = neighbors(rng, space, values(parm), step, n)
    end
    @debug "Finished hill-climbing"

    return parm
end

hc(f::Function, space::AbstractSpace, data::AbstractResampler, step::Tuple; args::NamedTuple = NamedTuple(), n::Int = 1, maximize::Bool = false) =
    hc(GLOBAL_RNG, f, space, data, step, args = args, n = n, maximize = maximize)

function hcfit(rng::AbstractRNG, f::Function, space::AbstractSpace, data::MonadicResampler, step::Tuple; args::NamedTuple = NamedTuple(), n::Int = 1, maximize::Bool = false)
    n ≥ 1 || throw(ArgumentError("invalid sample size of $n"))

    model = nothing
    best = maximize ? -Inf : Inf
  
    @debug "Start hill-climbing"
    nb = neighbors(rng, space, values(rand(rng, space)), step, n)
    @inbounds while !isempty(nb)
        models, loss = _fit_split(f, nb, first(data)..., args)
        if maximize
            i = argmax(loss)
            loss[i] > best || break
        else
            i = argmin(loss)
            loss[i] < best || break
        end
        model, best = models[i], loss[i]
        nb = neighbors(rng, space, values(nb[i]), step, n)
    end
    @debug "Finished hill-climbing"

    return model
end

hcfit(f::Function, space::AbstractSpace, data::MonadicResampler, step::Tuple; args::NamedTuple = NamedTuple(), n::Int = 1, maximize::Bool = false) =
    hcfit(GLOBAL_RNG, f, space, data, step, args = args, n = n, maximize = maximize)

struct Budget{name, T<:Real}
    val::T
    function Budget{name}(val::Real) where name
        return new{name, typeof(val)}(val)
    end
end

_cast(::Type{T}, x::Real, r) where T<:Real = T(x)
_cast(::Type{T}, x::AbstractFloat, r) where T<:Integer = round(T, x, r)
_cast(::Type{T}, x::T, r) where T<:Real = x

abstract type AbstractAllocation end

struct GeometricAllocation <: AbstractAllocation
    rate::Float64
    function GeometricAllocation(rate::Real)
        rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))
        return new(rate)
    end
end

struct ContstantAllocation <: AbstractAllocation
    rate::Float64
    function ContstantAllocation(rate::Real)
        rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))
        return new(rate)
    end
end

struct HyperbandAllocation <: AbstractAllocation
    n::Int
    rate::Float64
    function HyperbandAllocation(n::Int, rate::Real)
        n > 0 || throw(ArgumentError("unable to allocate arms over $n rounds"))
        rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))
        return new(n, rate)
    end
end

@propagate_inbounds function allocate(budget::Budget{name, T}, mode::GeometricAllocation, narms::Int) where {name, T}
    n = floor(Int, log(mode.rate, narms)) + 1
    arms = Vector{Int}(undef, n)
    args = Vector{NamedTuple{(name,), Tuple{T}}}(undef, n)
    for i in OneTo(n)
        c = 1 / (round(Int, narms / mode.rate^(i - 1)) * n)
        args[i] = NamedTuple{(name,)}(_cast(T, c * budget.val, RoundDown))
        arms[i] = ceil(Int, narms / mode.rate^i)
    end
    return zip(arms, args)
end

@propagate_inbounds function allocate(budget::Budget{name, T}, mode::ContstantAllocation, narms::Int) where {name, T}
    n = floor(Int, log(mode.rate, narms)) + 1
    arms = Vector{Int}(undef, n)
    args = Vector{NamedTuple{(name,), Tuple{T}}}(undef, n)
    c = (mode.rate - 1) * mode.rate^(n - 1) / (narms * (mode.rate^n - 1))
    for i in OneTo(n)
        args[i] = NamedTuple{(name,)}(_cast(T, c * budget.val, RoundDown))
        arms[i] = ceil(Int, narms / mode.rate^i)
    end
    return zip(arms, args)
end

@propagate_inbounds function allocate(budget::Budget{name, T}, mode::HyperbandAllocation, narms::Int) where {name, T}
    arms = Vector{Int}(undef, mode.n)
    args = Vector{NamedTuple{(name,), Tuple{T}}}(undef, mode.n)
    for i in OneTo(mode.n)
        c = 1 / mode.rate^(mode.n - i)
        args[i] = NamedTuple{(name,)}(_cast(T, c * budget.val, RoundNearest)) #RoundNearest?
        arms[i] = max(floor(Int, narms / mode.rate^i), 1)
    end
    return zip(arms, args)
end

@inline function _sha(f, space, data, budget, args, mode, maximize)
    length(space) ≥ 1 || throw(ArgumentError("nothing to optimize"))

    loss = nothing
    arms = map(f, space)
    train, val = first(data)

    @debug "Start successive halving"
    @inbounds for (k, arg) in allocate(budget, mode, length(arms))
        args = merge(args, arg)
        arms = pmap(x -> _fit!(x, train, args), arms)
        loss = map(x -> _loss(x, val), arms)
        @debug "Fitted arms" space args loss
        inds = sortperm(loss, rev = maximize)[OneTo(k)]
        arms, space = arms[inds], space[inds]
    end
    @debug "Finished successive halving"

    return first(arms), first(space), first(loss)
end

sha(f::Function, space::Union{FiniteSpace, Vector{T}}, data::MonadicResampler, budget::Budget; args::NamedTuple = NamedTuple(), mode::AbstractAllocation = GeometricAllocation(2), maximize::Bool = false) where T<:NamedTuple =
    _sha(f, space, data, budget, args, mode, maximize)[2]

shafit(f::Function, space::Union{FiniteSpace, Vector{T}}, data::MonadicResampler, budget::Budget; args::NamedTuple = NamedTuple(), mode::AbstractAllocation = GeometricAllocation(2), maximize::Bool = false) where T<:NamedTuple =
    _sha(f, space, data, budget, args, mode, maximize)[1]

@inline function _hyperband(rng, f, space, data, budget, args, rate, maximize)
    budget.val ≥ 1 || throw(ArgumentError("insufficient budget to allocate brackets"))
    rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))

    n = floor(Int, log(rate, budget.val)) + 1

    @debug "Start hyperband"
    best = _sha(f, rand(rng, space, ceil(Int, rate^(n - 1))), 
                data, budget, args, HyperbandAllocation(n, rate), maximize)

    @inbounds for i in reverse(OneTo(n - 1))
        curr = _sha(f, rand(rng, space, ceil(Int, n * rate^(i - 1) / i)), 
                    data, budget, args, HyperbandAllocation(i, rate), maximize)

        if maximize
            curr[3] > best[3] || continue
        else
            curr[3] < best[3] || continue
        end

        best = curr
    end
    @debug "Finished hyperband"

    return best
end

hyperband(rng::AbstractRNG, f::Function, space::AbstractSpace, data::MonadicResampler, budget::Budget; args::NamedTuple = NamedTuple(), rate::Real = 3, maximize::Bool = false) =
    _hyperband(rng, f, space, data, budget, args, rate, maximize)[2]
hyperband(f::Function, space::AbstractSpace, data::MonadicResampler, budget::Budget; args::NamedTuple = NamedTuple(), rate::Real = 3, maximize::Bool = false) =
    hyperband(GLOBAL_RNG, f, space, data, budget, args = args, rate = rate, maximize = maximize)

hyperbandfit(rng::AbstractRNG, f::Function, space::AbstractSpace, data::MonadicResampler, budget::Budget; args::NamedTuple = NamedTuple(), rate::Real = 3, maximize::Bool = false) =
    _hyperband(rng, f, space, data, budget, args, rate, maximize)[1]
hyperbandfit(f::Function, space::AbstractSpace, data::MonadicResampler, budget::Budget; args::NamedTuple = NamedTuple(), rate::Real = 3, maximize::Bool = false) =
    hyperbandfit(GLOBAL_RNG, f, space, data, budget, args = args, rate = rate, maximize = maximize)

@inline function _sasha(rng, f, space, data, args, temp, maximize)
    length(space) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    temp ≥ 0 || throw(ArgumentError("initial temperature must be positive"))

    train, test = first(data)
    arms = map(f, space)

    n = 1
    @debug "Start SASHA"
    @inbounds while length(arms) > 1
        arms = pmap(x -> _fit!(x, train, args), arms)
        loss = map(x -> _loss(x, test), arms)

        if maximize
            prob = exp.(n .* (loss .- max(loss...)) ./ temp)
        else
            prob = exp.(-n .* (loss .- min(loss...)) ./ temp)
        end        

        @debug "Fitted arms" space prob loss

        inds = findall(rand(rng, length(prob)) .≤ prob)
        arms, space = arms[inds], space[inds]

        n += 1
    end
    @debug "Finished SASHA"

    return first(arms), first(space)
end

sasha(rng::AbstractRNG, f::Function, space::Union{FiniteSpace, Vector{T}}, data::MonadicResampler; args::NamedTuple = NamedTuple(), temp::Real = 1, maximize::Bool = false) where T<:NamedTuple =
    _sasha(rng, f, space, data, args, temp, maximize)[2]
sasha(f::Function, space::Union{FiniteSpace, Vector{T}}, data::MonadicResampler; args::NamedTuple = NamedTuple(), temp::Real = 1, maximize::Bool = false) where T<:NamedTuple =
    sasha(GLOBAL_RNG, f, space, data, args = args, temp = temp, maximize = maximize)

sashafit(rng::AbstractRNG, f::Function, space::Union{FiniteSpace, Vector{T}}, data::MonadicResampler; args::NamedTuple = NamedTuple(), temp::Real = 1, maximize::Bool = false) where T<:NamedTuple =
    _sasha(rng, f, space, data, args, temp, maximize)[1]
sashafit(f::Function, space::Union{FiniteSpace, Vector{T}}, data::MonadicResampler; args::NamedTuple = NamedTuple(), temp::Real = 1, maximize::Bool = false) where T<:NamedTuple =
    sashafit(GLOBAL_RNG, f, space, data, args = args, temp = temp, maximize = maximize)

end
