module CrossValidation

using Base: @propagate_inbounds, OneTo
using Random: GLOBAL_RNG, AbstractRNG, shuffle!
using Distributed: pmap

import Random: rand

export AbstractResampler, MonadicResampler, VariadicResampler, FixedSplit, RandomSplit, LeaveOneOut, KFold, ForwardChaining, SlidingWindow,
       AbstractSpace, FiniteSpace, InfiniteSpace, space,
       AbstractDistribution, DiscreteDistribution, ContinousDistribution, Discrete, DiscreteUniform, Uniform, LogUniform, Normal, sample,
       Budget, AllocationMode, GeometricAllocation, ConstantAllocation, HyperbandAllocation, allocate,
       fit!, loss, validate, brute, brute_fit, hc, hc_fit, sha, sha_fit, hyperband, hyperband_fit, sasha, sasha_fit

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

restype(x) = restype(typeof(x))
restype(x::Type{T}) where T<:AbstractRange = Vector{eltype(x)}
restype(x::Type{T}) where T = T

abstract type AbstractResampler end
abstract type MonadicResampler{D} <: AbstractResampler end
abstract type VariadicResampler{D} <: AbstractResampler end

Base.eltype(::Type{R}) where R<:MonadicResampler{D} where D = Tuple{restype(D), restype(D)}
Base.eltype(::Type{R}) where R<:VariadicResampler{D} where D = Tuple{restype(D), restype(D)}

struct FixedSplit{D} <: MonadicResampler{D}
    data::D
    m::Int
    function FixedSplit(data, m::Int)
        n = nobs(data)
        1 ≤ m < n || throw(ArgumentError("data cannot be split by $m"))
        return new{typeof(data)}(data, m)
    end
end

FixedSplit(data, ratio::Real) = FixedSplit(data, floor(Int, nobs(data) * ratio))

Base.length(r::FixedSplit) = 1

@propagate_inbounds function Base.iterate(r::FixedSplit, state = 1)
    state > 1 && return nothing
    x = getobs(r.data, OneTo(r.m))
    y = getobs(r.data, (r.m + 1):nobs(r.data))
    return (x, y), state + 1
end

struct RandomSplit{D} <: MonadicResampler{D}
    data::D
    m::Int
    perm::Vector{Int}
    function RandomSplit(data, m::Int)
        n = nobs(data)
        1 ≤ m < n || throw(ArgumentError("data cannot be split by $m"))
        return new{typeof(data)}(data, m, shuffle!([OneTo(n);]))
    end
end

RandomSplit(data, ratio::Real) = RandomSplit(data, floor(Int, nobs(data) * ratio))

Base.length(r::RandomSplit) = 1

@propagate_inbounds function Base.iterate(r::RandomSplit, state = 1)
    state > 1 && return nothing
    x = getobs(r.data, r.perm[OneTo(r.m)])
    y = getobs(r.data, r.perm[(r.m + 1):nobs(r.data)])
    return (x, y), state + 1
end

struct LeaveOneOut{D} <: VariadicResampler{D}
    data::D
    function LeaveOneOut(data)
        n = nobs(data)
        n > 1 || throw(ArgumentError("data has too few observations to split"))
        return new{typeof(data)}(data)
    end
end

Base.length(r::LeaveOneOut) = nobs(r.data)

@propagate_inbounds function Base.iterate(r::LeaveOneOut, state = 1)
    state > length(r) && return nothing
    x = getobs(r.data, union(OneTo(state - 1), (state + 1):nobs(r.data)))
    y = getobs(r.data, state:state)
    return (x, y), state + 1
end

struct KFold{D} <: VariadicResampler{D}
    data::D
    k::Int
    perm::Vector{Int}
    function KFold(data, k::Int)
        n = nobs(data)
        1 < k ≤ n || throw(ArgumentError("data cannot be partitioned into $k folds"))
        return new{typeof(data)}(data, k, shuffle!([OneTo(n);]))
    end
end

Base.length(r::KFold) = r.k

@propagate_inbounds function Base.iterate(r::KFold, state = 1)
    state > length(r) && return nothing
    n = nobs(r.data)
    m, w = mod(n, r.k), floor(Int, n / r.k)
    fold = ((state - 1) * w + min(m, state - 1) + 1):(state * w + min(m, state))
    x = getobs(r.data, r.perm[setdiff(OneTo(n), fold)])
    y = getobs(r.data, r.perm[fold])
    return (x, y), state + 1
end

struct ForwardChaining{D} <: VariadicResampler{D}
    data::D
    init::Int
    out::Int
    partial::Bool
    function ForwardChaining(data, init::Int, out::Int; partial::Bool = true)
        n = nobs(data)
        1 ≤ init ≤ n || throw(ArgumentError("invalid initial window of $init"))
        1 ≤ out ≤ n || throw(ArgumentError("invalid out-of-sample window of $out"))
        init + out ≤ n || throw(ArgumentError("initial and out-of-sample window exceed the number of data observations"))
        return new{typeof(data)}(data, init, out, partial)
    end
end

function Base.length(r::ForwardChaining)
    l = (nobs(r.data) - r.init) / r.out
    return r.partial ? ceil(Int, l) : floor(Int, l)
end

@propagate_inbounds function Base.iterate(r::ForwardChaining, state = 1)
    state > length(r) && return nothing
    x = getobs(r.data, OneTo(r.init + (state - 1) * r.out))
    y = getobs(r.data, (r.init + (state - 1) * r.out + 1):min(r.init + state * r.out, nobs(r.data)))
    return (x, y), state + 1
end

struct SlidingWindow{D} <: VariadicResampler{D}
    data::D
    window::Int
    out::Int
    partial::Bool
    function SlidingWindow(data, window::Int, out::Int; partial::Bool = true)
        n = nobs(data)
        1 ≤ window ≤ n || throw(ArgumentError("invalid sliding window of $window"))
        1 ≤ out ≤ n || throw(ArgumentError("invalid out-of-sample window of $out"))
        window + out ≤ n || throw(ArgumentError("sliding and out-of-sample window exceed the number of data observations"))
        return new{typeof(data)}(data, window, out, partial)
    end
end

function Base.length(r::SlidingWindow)
    l = (nobs(r.data) - r.window) / r.out
    return r.partial ? ceil(Int, l) : floor(Int, l)
end

@propagate_inbounds function Base.iterate(r::SlidingWindow, state = 1)
    state > length(r) && return nothing
    x = getobs(r.data, (1 + (state - 1) * r.out):(r.window + (state - 1) * r.out))
    y = getobs(r.data, (r.window + (state - 1) * r.out + 1):min(r.window + state * r.out, nobs(r.data)))
    return (x, y), state + 1
end

function sample(rng::AbstractRNG, s, n::Integer)
    m = length(s)
    1 ≤ n ≤ m || throw(ArgumentError("cannot sample $n times without replacement"))
    vals = sizehint!(eltype(s)[], n)
    for _ in OneTo(n)
        val = rand(rng, s)
        while val in vals
            val = rand(rng, s)
        end
        push!(vals, val)
    end
    return vals
end

sample(rng::AbstractRNG, s) = rand(rng, s)

sample(s, n) = sample(GLOBAL_RNG, s, n)
sample(s) = sample(GLOBAL_RNG, s)

abstract type AbstractDistribution end
abstract type DiscreteDistribution{V} <: AbstractDistribution end
abstract type ContinousDistribution{S} <: AbstractDistribution end

Base.eltype(::Type{D}) where D<:DiscreteDistribution{V} where V = eltype(V)
Base.eltype(::Type{D}) where D<:ContinousDistribution{S} where S = S

Base.length(d::DiscreteDistribution) = length(values(d))
Base.getindex(d::DiscreteDistribution, i) = getindex(values(d), i)
Base.iterate(d::DiscreteDistribution) = iterate(values(d))
Base.iterate(d::DiscreteDistribution, state) = iterate(values(d), state)

struct Discrete{V, P<:AbstractFloat} <: DiscreteDistribution{V}
    vals::V
    probs::Vector{P}
    function Discrete(vals::V, probs::Vector{P}) where {V, P<:AbstractFloat}
        length(vals) == length(probs) || throw(ArgumentError("lenghts of values and probabilities do not match"))
        (all(probs .≥ 0) && isapprox(sum(probs), 1)) || throw(ArgumentError("invalid probabilities provided"))
        return new{V, P}(vals, probs)
    end
end

Base.values(d::Discrete) = d.vals

function rand(rng::AbstractRNG, d::Discrete{V, P}) where {V, P}
    c = zero(P)
    q = rand(rng)
    for (state, p) in zip(d.vals, d.probs)
        c += p
        if q < c
            return state
        end
    end
    throw(ErrorException("could not generate random element from distribution"))
end

struct DiscreteUniform{V} <: DiscreteDistribution{V}
    vals::V
end

Base.values(d::DiscreteUniform) = d.vals

rand(rng::AbstractRNG, d::DiscreteUniform) = rand(rng, d.vals)

struct Uniform{S<:AbstractFloat, P<:Real} <: ContinousDistribution{S}
    a::P
    b::P
    function Uniform{S}(a::Real, b::Real) where S<:AbstractFloat
        a < b || throw(ArgumentError("a must be smaller than b"))
        a, b = promote(a, b)
        return new{S, typeof(a)}(a, b)
    end
end

Uniform(a::Real, b::Real) = Uniform{Float64}(a, b)

rand(rng::AbstractRNG, d::Uniform{S, P}) where {S, P} = S(d.a + (d.b - d.a) * rand(rng, float(P)))

struct LogUniform{S<:AbstractFloat, P<:Real} <: ContinousDistribution{S}
    a::P
    b::P
    function LogUniform{S}(a::Real, b::Real) where S<:AbstractFloat
        a < b || throw(ArgumentError("a must be smaller than b"))
        a, b = promote(a, b)
        return new{S, typeof(a)}(a, b)
    end
end

LogUniform(a::Real, b::Real) = LogUniform{Float64}(a, b)

rand(rng::AbstractRNG, d::LogUniform{S, P}) where {S, P} = S(exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng, float(P))))

struct Normal{S<:AbstractFloat, P<:Real} <: ContinousDistribution{S}
    mean::P
    std::P
    function Normal{S}(mean::Real, std::Real) where S<:AbstractFloat
        std > zero(std) || throw(ArgumentError("standard deviation must be larger than zero"))
        mean, std = promote(mean, std)
        return new{S, typeof(mean)}(mean, std)
    end
end

Normal(mean::Real, std::Real) = Normal{Float64}(mean, std)

rand(rng::AbstractRNG, d::Normal{S, P}) where {S, P} = S(d.mean + d.std * randn(rng, float(P)))

abstract type AbstractSpace end

struct FiniteSpace{names, T<:Tuple} <: AbstractSpace
    vars::T
end

Base.eltype(::Type{S}) where S<:FiniteSpace{names, T} where {names, T} = NamedTuple{names, Tuple{map(eltype, T.parameters)...}}

Base.length(s::FiniteSpace) = length(s.vars) == 0 ? 0 : prod(length, s.vars)
Base.keys(s::FiniteSpace) = OneTo(length(s))
Base.firstindex(s::FiniteSpace) = 1
Base.lastindex(s::FiniteSpace) = length(s)

Base.size(s::FiniteSpace) = length(s.vars) == 0 ? (0,) : map(length, s.vars)

@inline function Base.getindex(s::FiniteSpace{names, T}, i::Int) where {names, T}
    @boundscheck 1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    strides = (1, cumprod(map(length, Base.front(s.vars)))...)
    return NamedTuple{names}(map(getindex, s.vars, mod.((i - 1) .÷ strides, size(s)) .+ 1))
end

@inline function Base.getindex(s::FiniteSpace{names, T}, I::Vararg{Int, N}) where {names, T, N}
    @boundscheck length(I) == length(s.vars) && all(1 .≤ I .≤ size(s)) || throw(BoundsError(s, I))
    return NamedTuple{names}(map(getindex, s.vars, I))
end

@inline function Base.getindex(s::FiniteSpace{names, T}, inds::Vector{Int}) where {names, T}
    return [s[i] for i in inds]
end

@propagate_inbounds function Base.iterate(s::FiniteSpace, state = 1)
    state > length(s) && return nothing
    return s[state], state + 1
end

rand(rng::AbstractRNG, s::FiniteSpace{names}) where {names} = NamedTuple{names}(map(x -> rand(rng, x), s.vars))
sample(rng::AbstractRNG, s::FiniteSpace, n::Int) = [s[i] for i in sample(rng, OneTo(length(s)), n)]

struct InfiniteSpace{names, T<:Tuple} <: AbstractSpace
    vars::T
end

Base.eltype(::Type{S}) where S<:InfiniteSpace{names, T} where {names, T} = NamedTuple{names, Tuple{map(eltype, T.parameters)...}}

rand(rng::AbstractRNG, s::InfiniteSpace{names}) where {names} = NamedTuple{names}(map(x -> rand(rng, x), s.vars))
sample(rng::AbstractRNG, s::InfiniteSpace, n::Int) = [rand(rng, s) for _ in OneTo(n)]

space(; vars...) = space(keys(vars), values(values(vars)))
space(names, vars::Tuple{Vararg{DiscreteDistribution}}) = FiniteSpace{names, typeof(vars)}(vars)
space(names, vars::Tuple{Vararg{AbstractDistribution}}) = InfiniteSpace{names, typeof(vars)}(vars)

_fit!(model, x::Union{Tuple, NamedTuple}, args) = fit!(model, x...; args...)
_fit!(model, x, args) = fit!(model, x; args...)

fit!(model, x) = throw(MethodError(fit!, (model, x)))

_loss(model, x::Union{Tuple, NamedTuple}) = loss(model, x...)
_loss(model, x) = loss(model, x)

loss(model, x) = throw(MethodError(loss, (model, x)))

@inline function _val(T, parms, data, args)
    return sum(x -> _val_split(T, parms, x..., args), data) / length(data)
end

@inline function _val_split(T, parms, train, test, args)
    models = pmap(x -> _fit!(T(; x...), train, args), parms)
    loss = map(x -> _loss(x, test), models)
    @debug "Fitted models" parms args loss
    return loss
end

@inline function _fit_split(T, parms, train, test, args)
    models = pmap(x -> _fit!(T(; x...), train, args), parms)
    loss = map(x -> _loss(x, test), models)
    @debug "Fitted models" parms args loss
    return models, loss
end

function validate(model, data::AbstractResampler; args::NamedTuple = ())
    @debug "Start model validation"
    loss = map(x -> _loss(_fit!(model, x[1], args), x[2]), data)
    @debug "Finished model validation"
    return loss
end

function validate(f::Function, data::AbstractResampler)
    @debug "Start model validation"
    loss = map(x -> _loss(f(x[1]), x[2]), data)
    @debug "Finished model validation"
    return loss
end

function brute(T::Type, parms, data::AbstractResampler; args = (), maximize::Bool = false)
    length(parms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    
    @debug "Start brute-force search"
    loss = _val(T, parms, data, args)
    ind = maximize ? argmax(loss) : argmin(loss)
    @debug "Finished brute-force search"
    
    return parms[ind]
end

function brute_fit(T::Type, parms, data::MonadicResampler; args = (), maximize::Bool = false)
    length(parms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    
    train, val = first(data)

    @debug "Start brute-force search"
    models, loss = _fit_split(T, parms, train, val, args)
    ind = maximize ? argmax(loss) : argmin(loss)
    @debug "Finished brute-force search"
    
    return models[ind]
end

function _neighbors(space, ref, k, bl)
    dim = size(space)
    inds = sizehint!(Int[], sum(min.(dim .- 1, 2 * k)))
    @inbounds for i in eachindex(dim)
        if i == 1
            d = mod(ref - 1, dim[1]) + 1
            for j in reverse(OneTo(k))
                if d - j ≥ 1
                    ind = ref - j
                    if ind ∉ bl
                        push!(inds, ind)
                    end
                end
            end
            for j in OneTo(k)
                if d + j ≤ dim[1]
                    ind = ref + j
                    if ind ∉ bl
                        push!(inds, ind)
                    end
                end
            end
        else
            d = mod((ref - 1) ÷ dim[i - 1], dim[i]) + 1
            for j in reverse(OneTo(k))
                if d - j ≥ 1
                    ind = ref - j * dim[i - 1]
                    if ind ∉ bl
                        push!(inds, ind)
                    end
                end
            end
            for j in OneTo(k)
                if d + j ≤ dim[i]
                    ind = ref + j * dim[i - 1]
                    if ind ∉ bl
                        push!(inds, ind)
                    end
                end
            end
        end
    end
    return inds
end

function hc(T::Type, space::FiniteSpace, data::AbstractResampler; args = (), nstart::Int = 1, k::Int = 1, maximize::Bool = false)
    length(space) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    k ≥ 1 || throw(ArgumentError("invalid neighborhood size of $k"))

    bl = Int[]
    parm = nothing
    best = maximize ? -Inf : Inf

    cand = sample(OneTo(length(space)), nstart)

    @debug "Start hill-climbing"
    while !isempty(cand)
        append!(bl, cand)

        parms = space[cand]
        loss = _val(T, parms, data, args)

        if maximize
            i = argmax(loss)
            loss[i] > best || break
        else
            i = argmin(loss)
            loss[i] < best || break
        end

        parm, best = parms[i], loss[i]
        cand = _neighbors(space, cand[i], k, bl)
    end
    @debug "Finished hill-climbing"

    return parm
end

function hc_fit(T::Type, space::FiniteSpace, data::MonadicResampler; args = (), nstart::Int = 1, k::Int = 1, maximize::Bool = false)
    length(space) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    k ≥ 1 || throw(ArgumentError("invalid neighborhood size of $k"))

    bl = Int[]
    model = nothing
    best = maximize ? -Inf : Inf

    train, val = first(data)
    cand = sample(OneTo(length(space)), nstart)

    @debug "Start hill-climbing"
    while !isempty(cand)
        append!(bl, cand)

        models, loss = _fit_split(T, space[cand], train, val, args)

        if maximize
            i = argmax(loss)
            loss[i] > best || break
        else
            i = argmin(loss)
            loss[i] < best || break
        end

        model, best = models[i], loss[i]
        cand = _neighbors(space, cand[i], k, bl)
    end
    @debug "Finished hill-climbing"

    return model
end

struct Budget{name, T<:Real}
    val::T
    function Budget{name}(val::Real) where name
        return new{name, typeof(val)}(val)
    end
end

_cast(::Type{T}, x::Real, r) where T <: Real = T(x)
_cast(::Type{T}, x::AbstractFloat, r) where T <: Integer = round(T, x, r)
_cast(::Type{T}, x::T, r) where T <: Real = x

struct AllocationMode{M} end

const GeometricAllocation = AllocationMode{:Geometric}()
const ConstantAllocation = AllocationMode{:Constant}()
const HyperbandAllocation = AllocationMode{:Hyperband}()

function allocate(budget::Budget, mode::AllocationMode, narms::Int, rate::Real)
    nrounds = floor(Int, log(rate, narms)) + 1
    return allocate(budget, mode, nrounds, narms, rate)
end

function allocate(budget::Budget{name, T}, mode::AllocationMode{:Geometric}, nrounds::Int, narms::Int, rate::Real) where {name, T}
    arms = Vector{Int}(undef, nrounds)
    args = Vector{NamedTuple{(name,), Tuple{T}}}(undef, nrounds)
    for i in OneTo(nrounds)
        c = 1 / (round(Int, narms / rate^(i - 1)) * nrounds)
        args[i] = NamedTuple{(name,)}(_cast(typeof(budget.val), c * budget.val, RoundDown))
        arms[i] = ceil(Int, narms / rate^i)
    end
    return zip(arms, args)
end

function allocate(budget::Budget{name, T}, mode::AllocationMode{:Constant}, nrounds::Int, narms::Int, rate::Real) where {name, T}
    arms = Vector{Int}(undef, nrounds)
    args = Vector{NamedTuple{(name,), Tuple{T}}}(undef, nrounds)
    c = (rate - 1) * rate^(nrounds - 1) / (narms * (rate^nrounds - 1))
    for i in OneTo(nrounds)
        args[i] = NamedTuple{(name,)}(_cast(typeof(budget.val), c * budget.val, RoundDown))
        arms[i] = ceil(Int, narms / rate^i)
    end
    return zip(arms, args)
end

function allocate(budget::Budget{name, T}, mode::AllocationMode{:Hyperband}, nrounds::Int, narms::Int, rate::Real) where {name, T}
    arms = Vector{Int}(undef, nrounds)
    args = Vector{NamedTuple{(name,), Tuple{T}}}(undef, nrounds)
    for i in OneTo(nrounds)
        c = 1 / rate^(nrounds - i)
        args[i] = NamedTuple{(name,)}(_cast(typeof(budget.val), c * budget.val, RoundNearest)) #RoundNearest?
        arms[i] = max(floor(Int, narms / rate^i), 1)
    end
    return zip(arms, args)
end

@inline function _sha(T, parms, data, budget, mode, rate, maximize)
    length(parms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))

    train, val = first(data)
    arms = map(x -> T(; x...), parms)

    @debug "Start successive halving"
    for (k, args) in allocate(budget, mode, length(arms), rate)
        arms = pmap(x -> _fit!(x, train, args), arms)
        loss = map(x -> _loss(x, val), arms)
        @debug "Fitted arms" parms args loss
        inds = sortperm(loss, rev=maximize)[OneTo(k)]
        arms, parms = arms[inds], parms[inds]
    end
    @debug "Finished successive halving"

    return first(arms), first(parms)
end

sha(T::Type, parms, data::MonadicResampler, budget::Budget; mode::AllocationMode = GeometricAllocation, rate::Real = 2, maximize::Bool = false) =
    _sha(T, parms, data, budget, mode, rate, maximize)[2]

sha_fit(T::Type, parms, data::MonadicResampler, budget::Budget; mode::AllocationMode = GeometricAllocation, rate::Real = 2, maximize::Bool = false) =
    _sha(T, parms, data, budget, mode, rate, maximize)[1]

@inline function _hyperband(T, space, data, budget, rate, maximize)
    rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))

    arm, parm = nothing, nothing
    best = maximize ? -Inf : Inf

    train, val = first(data)
    n = floor(Int, log(rate, budget.val)) + 1

    @debug "Start hyperband"
    for i in reverse(OneTo(n))
        narms = ceil(Int, n * rate^(i - 1) / i)

        loss = nothing
        parms = sample(space, narms)
        arms = map(x -> T(; x...), parms)

        @debug "Start successive halving"
        for (k, args) in allocate(budget, HyperbandAllocation, i, narms, rate)
            arms = pmap(x -> _fit!(x, train, args), arms)
            loss = map(x -> _loss(x, val), arms)
            @debug "Fitted arms" parms args loss
            inds = sortperm(loss, rev=maximize)[OneTo(k)]
            arms, parms = arms[inds], parms[inds]
        end
        @debug "Finished successive halving"

        if maximize
            first(loss) > best || continue
        else
            first(loss) < best || continue
        end

        arm, parm = first(arms), first(parms)
        best = first(loss)
    end
    @debug "Finished hyperband"

    return arm, parm
end

hyperband(T::Type, space::AbstractSpace, data::MonadicResampler, budget::Budget; rate::Real = 3, maximize::Bool = false) =
    _hyperband(T, space, data, budget, rate, maximize)[2]
hyperband_fit(T::Type, space::AbstractSpace, data::MonadicResampler, budget::Budget; rate::Real = 3, maximize::Bool = false) =
    _hyperband(T, space, data, budget, rate, maximize)[1]
 
@inline function _sasha(T, parms, data, args, temp, maximize)
    length(parms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    temp ≥ 0 || throw(ArgumentError("initial temperature must be positive"))

    train, test = first(data)
    arms = map(x -> T(; x...), parms)

    n = 1
    @debug "Start SASHA"
    while length(arms) > 1
        arms = pmap(x -> _fit!(x, train, args), arms)
        loss = map(x -> _loss(x, test), arms)

        if maximize
            prob = exp.(n .* (loss .- max(loss...)) ./ temp)
        else
            prob = exp.(-n .* (loss .- min(loss...)) ./ temp)
        end        

        @debug "Fitted arms" parms prob loss

        inds = findall(rand(length(prob)) .≤ prob)
        arms, parms = arms[inds], parms[inds]

        n += 1
    end
    @debug "Finished SASHA"

    return first(arms), first(parms)
end

sasha(T::Type, parms, data::MonadicResampler; args = (), temp::Real = 1, maximize::Bool = false) =
    _sasha(T, parms, data, args, temp, maximize)[2]

sasha_fit(T::Type, parms, data::MonadicResampler; args = (), temp::Real = 1, maximize::Bool = false) =
    _sasha(T, parms, data, args, temp, maximize)[1]

end