module CrossValidation

using Base: @propagate_inbounds, OneTo
using Random: GLOBAL_RNG, AbstractRNG, shuffle!
using Distributed: pmap

import Random: rand

export AbstractResampler, FixedSplit, RandomSplit, LeaveOneOut, KFold, ForwardChaining, SlidingWindow,
       AbstractSpace, FiniteSpace, InfiniteSpace, space, ParameterVector,
       AbstractDistribution, DiscreteDistribution, ContinousDistribution, Discrete, DiscreteUniform, Uniform, LogUniform, Normal, sample,
       Budget, AllocationMode, GeometricAllocation, ConstantAllocation, HyperbandAllocation, allocate,
       fit!, loss, validate, brute, hc, sha, hyperband, sasha

nobs(x) = length(x)
nobs(x::AbstractArray) = size(x)[end]

function nobs(x::Union{Tuple, NamedTuple})
    length(x) > 0 || return 0
    n = nobs(first(x))
    if !all(y -> nobs(y) == n, Base.tail(x))
        throw(ArgumentError("all data should have the same number of observations"))
    end
    return n
end

getobs(x::AbstractArray, i) = x[Base.setindex(ntuple(x -> Colon(), ndims(x)), i, ndims(x))...]
getobs(x::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(getobs, i), x)

restype(x) = restype(typeof(x))
restype(x::Type{T}) where T<:AbstractRange = Vector{eltype(x)}
restype(x::Type{T}) where T = T

abstract type AbstractResampler end

Base.eltype(r::AbstractResampler) = Tuple{restype(r.data), restype(r.data)}

struct FixedSplit{D} <: AbstractResampler
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
    train = getobs(r.data, OneTo(r.m))
    test = getobs(r.data, (r.m + 1):nobs(r.data))
    return (train, test), state + 1
end

struct RandomSplit{D} <: AbstractResampler
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
    train = getobs(r.data, r.perm[OneTo(r.m)])
    test = getobs(r.data, r.perm[(r.m + 1):nobs(r.data)])
    return (train, test), state + 1
end

struct LeaveOneOut{D} <: AbstractResampler
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
    train = getobs(r.data, union(OneTo(state - 1), (state + 1):nobs(r.data)))
    test = getobs(r.data, state:state)
    return (train, test), state + 1
end

struct KFold{D} <: AbstractResampler
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
    m = mod(n, r.k)
    w = floor(Int, n / r.k)
    fold = ((state - 1) * w + min(m, state - 1) + 1):(state * w + min(m, state))
    train = getobs(r.data, r.perm[setdiff(OneTo(n), fold)])
    test = getobs(r.data, r.perm[fold])
    return (train, test), state + 1
end

struct ForwardChaining{D} <: AbstractResampler
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
    train = getobs(r.data, OneTo(r.init + (state - 1) * r.out))
    test = getobs(r.data, (r.init + (state - 1) * r.out + 1):min(r.init + state * r.out, nobs(r.data)))
    return (train, test), state + 1
end

struct SlidingWindow{D} <: AbstractResampler
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
    train = getobs(r.data, (1 + (state - 1) * r.out):(r.window + (state - 1) * r.out))
    test = getobs(r.data, (r.window + (state - 1) * r.out + 1):min(r.window + state * r.out, nobs(r.data)))
    return (train, test), state + 1
end

function sample(rng::AbstractRNG, iter, n::Integer)
    m = length(iter)
    1 ≤ n ≤ m || throw(ArgumentError("cannot sample $n times without replacement"))
    vals = sizehint!(eltype(iter)[], n)
    for _ in OneTo(n)
        val = rand(rng, iter)
        while val in vals
            val = rand(rng, iter)
        end
        push!(vals, val)
    end
    return vals
end

sample(iter, n) = sample(GLOBAL_RNG, iter, n)
sample(iter) = sample(iter, 1)

abstract type AbstractDistribution end
abstract type DiscreteDistribution <: AbstractDistribution end
abstract type ContinousDistribution <: AbstractDistribution end

Base.eltype(d::DiscreteDistribution) = eltype(values(d))
Base.length(d::DiscreteDistribution) = length(values(d))
Base.getindex(d::DiscreteDistribution, i) = getindex(values(d), i)
Base.iterate(d::DiscreteDistribution) = Base.iterate(values(d))
Base.iterate(d::DiscreteDistribution, state) = Base.iterate(values(d), state)

struct Discrete{S, P<:AbstractFloat} <: DiscreteDistribution
    vals::S
    probs::Vector{P}
    function Discrete(states::S, probs::Vector{P}) where {S, P<:AbstractFloat}
        length(states) == length(probs) || throw(ArgumentError("lenghts of states and probabilities do not match"))
        (all(probs .≥ 0) && isapprox(sum(probs), 1)) || throw(ArgumentError("invalid probabilities provided"))
        return new{S, P}(states, probs)
    end
end

Base.values(d::Discrete) = d.vals

function rand(rng::AbstractRNG, d::Discrete{S, P}) where {S, P}
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

struct DiscreteUniform{S} <: DiscreteDistribution
    vals::S
end

Base.values(d::DiscreteUniform) = d.vals

rand(rng::AbstractRNG, d::DiscreteUniform) = rand(rng, d.vals)

struct Uniform{S<:Real, P<:Real} <: ContinousDistribution
    a::P
    b::P
    function Uniform{S}(a::Real, b::Real) where S<:Real
        a < b || throw(ArgumentError("a must be smaller than b"))
        a, b = promote(a, b)
        return new{S, typeof(a)}(a, b)
    end
end

Uniform(a::Real, b::Real) = Uniform{Float64}(a, b)

rand(rng::AbstractRNG, d::Uniform{S, P}) where {S, P} = S(d.a + (d.b - d.a) * rand(rng, float(P)))
rand(rng::AbstractRNG, d::Uniform{S, P}) where {S<:Unsigned, P} = round(S, abs(d.a + (d.b - d.a) * rand(rng, float(P))))
rand(rng::AbstractRNG, d::Uniform{S, P}) where {S<:Signed, P} = round(S, d.a + (d.b - d.a) * rand(rng, float(P)))
rand(rng::AbstractRNG, d::Uniform{S, P}) where {S<:Bool, P} = S(d.a + (d.b - d.a) * rand(rng, float(P)) ≥ 0.5)

struct LogUniform{S<:Real, P<:Real} <: ContinousDistribution
    a::P
    b::P
    function LogUniform{S}(a::Real, b::Real) where S<:Real
        a < b || throw(ArgumentError("a must be smaller than b"))
        a, b = promote(a, b)
        return new{S, typeof(a)}(a, b)
    end
end

LogUniform(a::Real, b::Real) = LogUniform{Float64}(a, b)

rand(rng::AbstractRNG, d::LogUniform{S, P}) where {S, P} = S(exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng, float(P))))
rand(rng::AbstractRNG, d::LogUniform{S, P}) where {S<:Unsigned, P} = round(S, abs(exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng, float(P)))))
rand(rng::AbstractRNG, d::LogUniform{S, P}) where {S<:Signed, P} = round(S, exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng, float(P))))
rand(rng::AbstractRNG, d::LogUniform{S, P}) where {S<:Bool, P} = S(exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng, float(P))) ≥ 0.5)

struct Normal{S<:Real, P<:Real} <: ContinousDistribution
    mean::P
    std::P
    function Normal{S}(mean::Real, std::Real) where S<:Real
        std > zero(std) || throw(ArgumentError("standard deviation must be larger than zero"))
        mean, std = promote(mean, std)
        return new{S, typeof(mean)}(mean, std)
    end
end

Normal(mean::Real, std::Real) = Normal{Float64}(mean, std)

rand(rng::AbstractRNG, d::Normal{S, P}) where {S, P} = S(d.mean + d.std * randn(rng, float(P)))
rand(rng::AbstractRNG, d::Normal{S, P}) where {S<:Unsigned, P} = round(S, abs(d.mean + d.std * randn(rng, float(P))))
rand(rng::AbstractRNG, d::Normal{S, P}) where {S<:Signed, P} = round(S, d.mean + d.std * randn(rng, float(P)))
rand(rng::AbstractRNG, d::Normal{S, P}) where {S<:Bool, P} = S(d.mean + d.std * randn(rng, float(P)) ≥ 0.5)

abstract type AbstractSpace end

struct FiniteSpace{names, T<:Tuple} <: AbstractSpace
    vars::T
end

Base.eltype(s::FiniteSpace{names, T}) where {names, T} = NamedTuple{names, Tuple{map(eltype, s.vars)...}}
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

rand(rng::AbstractRNG, space::FiniteSpace{names}) where {names} = NamedTuple{names}(map(x -> rand(rng, x), space.vars))

sample(rng::AbstractRNG, space::FiniteSpace) = rand(rng, space)
sample(rng::AbstractRNG, space::FiniteSpace, n::Int) = [space[i] for i in sample(rng, OneTo(length(space)), n)]

struct InfiniteSpace{names, T<:Tuple} <: AbstractSpace
    vars::T
end

rand(rng::AbstractRNG, space::InfiniteSpace{names}) where {names} = NamedTuple{names}(map(x -> rand(rng, x), space.vars))

sample(rng::AbstractRNG, space::InfiniteSpace) = rand(rng, space)
sample(rng::AbstractRNG, space::InfiniteSpace, n::Int) = [sample(rng, space) for _ in OneTo(n)]

space(; vars...) = space(keys(vars), values(values(vars)))
space(names, vars::Tuple{Vararg{DiscreteDistribution}}) = FiniteSpace{names, typeof(vars)}(vars)
space(names, vars::Tuple{Vararg{AbstractDistribution}}) = InfiniteSpace{names, typeof(vars)}(vars)

const ParameterVector = Array{NamedTuple{names, T}, 1} where {names, T}

_fit!(model, x::AbstractArray, args) = fit!(model, x; args...)
_fit!(model, x::Union{Tuple, NamedTuple}, args) = fit!(model, x...; args...)

fit!(model, x) = throw(MethodError(fit!, (model, x)))

_loss(model, x::AbstractArray) = loss(model, x)
_loss(model, x::Union{Tuple, NamedTuple}) = loss(model, x...)

loss(model, x) = throw(MethodError(loss, (model, x)))

@inline function _val(T, prms, data, args)
    return sum(x -> _val_split(T, prms, x..., args), data) / length(data)
end

function _val_split(T, prms, train, test, args)
    models = pmap(x -> _fit!(T(; x...), train, args), prms)
    loss = map(x -> _loss(x, test), models)
    @debug "Validated models" prms args loss
    return loss
end

function validate(model, data::AbstractResampler; args::NamedTuple = ())
    @debug "Start model validation"
    loss = map(x -> _loss(_fit!(model, x[1], args), x[2]), data)
    @debug "Finished model validation"
    return loss
end

_f(f, x::AbstractArray) = f(x)
_f(f, x::Union{Tuple, NamedTuple}) = f(x...)

function validate(f::Function, data::AbstractResampler)
    @debug "Start model validation"
    loss = map(x -> _loss(_f(f, x[1]), x[2]), data)
    @debug "Finished model validation"
    return loss
end

function brute(T::Type, prms::ParameterVector, data::AbstractResampler; args::NamedTuple = (), maximize::Bool = false)
    length(prms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    @debug "Start brute-force search"
    loss = _val(T, prms, data, args)
    ind = maximize ? argmax(loss) : argmin(loss)
    @debug "Finished brute-force search"
    return prms[ind]
end

brute(T::Type, space::FiniteSpace, data::AbstractResampler; args::NamedTuple = (), maximize::Bool = false) =
    brute(T, collect(space), data, args = args, maximize = maximize)

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

function hc(T::Type, space::FiniteSpace, data::AbstractResampler; args::NamedTuple = (), nstart::Int = 1, k::Int = 1, maximize::Bool = false)
    length(space) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    k ≥ 1 || throw(ArgumentError("invalid neighborhood size of $k"))

    bl = Int[]
    parm = nothing
    best = maximize ? -Inf : Inf

    cand = sample(OneTo(length(space)), nstart)

    @debug "Start hill-climbing"
    while !isempty(cand)
        append!(bl, cand)

        loss = _val(T, space[cand], data, args)
        if maximize
            i = argmax(loss)
            loss[i] > best || break
        else
            i = argmin(loss)
            loss[i] < best || break
        end

        parm = space[cand[i]]
        best = loss[i]

        cand = _neighbors(space, cand[i], k, bl)
    end
    @debug "Finished hill-climbing"

    return parm
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

struct AllocationMode{T} end

const GeometricAllocation = AllocationMode{:Geometric}()
const ConstantAllocation = AllocationMode{:Constant}()
const HyperbandAllocation = AllocationMode{:Hyperband}()

function allocate(budget::Budget, mode::AllocationMode{:Geometric}, narms::Int, rate::Real)
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

function allocate(budget::Budget, mode::AllocationMode{:Constant}, narms::Int, rate::Real)
    nrounds = floor(Int, log(rate, narms)) + 1
    return allocate(budget, mode, nrounds, narms, rate)
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

function allocate(budget::Budget, mode::AllocationMode{:Hyperband}, narms::Int, rate::Real)
    nrounds = floor(Int, log(rate, budget.val)) + 1
    return allocate(budget, mode, nrounds, narms, rate)
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

function sha(T::Type, prms::ParameterVector, data::AbstractResampler, budget::Budget; mode::AllocationMode = GeometricSchedule, rate::Number = 2, maximize::Bool = false)
    length(prms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    length(data) == 1 || throw(ArgumentError("can only optimize over one resample fold"))
    rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))

    train, test = first(data)
    arms = map(x -> T(; x...), prms)

    @debug "Start successive halving"
    for (k, args) in allocate(budget, mode, length(arms), rate)
        arms = pmap(x -> _fit!(x, train, args), arms)
        loss = map(x -> _loss(x, test), arms)
        @debug "Validated arms" prms args loss
        
        inds = sortperm(loss, rev=maximize)
        arms = arms[inds[OneTo(k)]]
        prms = prms[inds[OneTo(k)]]
    end
    @debug "Finished successive halving"

    return first(prms)
end

sha(T::Type, space::FiniteSpace, data::AbstractResampler, budget::Budget; mode::AllocationMode = GeometricSchedule, rate::Number = 2, maximize::Bool = false) =
    sha(T, collect(space), data, budget, mode = mode, rate = rate, maximize = maximize)

function hyperband(T::Type, space::AbstractSpace, data::AbstractResampler, budget::Budget; rate::Number = 3, maximize::Bool = false)
    length(data) == 1 || throw(ArgumentError("can only optimize over one resample fold"))
    rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))

    parm = nothing
    best = maximize ? -Inf : Inf

    train, test = first(data)
    n = floor(Int, log(rate, first(budget.val))) + 1

    @debug "Start hyperband"
    for i in reverse(OneTo(n))
        narms = ceil(Int, n * rate^(i - 1) / i)

        loss = nothing
        prms = sample(space, narms)
        arms = map(x -> T(; x...), prms)

        @debug "Start successive halving"
        for (k, args) in allocate(budget, HyperbandAllocation, i, narms, rate)
            arms = pmap(x -> _fit!(x, train, args), arms)
            loss = map(x -> _loss(x, test), arms)
            @debug "Validated arms" prms args loss
    
            inds = sortperm(loss, rev=maximize)
            arms = arms[inds[OneTo(k)]]
            prms = prms[inds[OneTo(k)]]    
        end
        @debug "Finished successive halving"

        if maximize
            first(loss) > best || continue
        else
            first(loss) < best || continue
        end

        parm = first(prms)
        best = first(loss)
    end
    @debug "Finished hyperband"

    return parm
end

function sasha(T::Type, prms::ParameterVector, data::AbstractResampler; args::NamedTuple = (), temp::Real = 1, maximize::Bool = false)
    length(prms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    length(data) == 1 || throw(ArgumentError("can only optimize over one resample fold"))
    temp ≥ 0 || throw(ArgumentError("initial temperature must be positive"))

    train, test = first(data)
    arms = map(x -> T(; x...), prms)

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

        @debug "Validated arms" prms prob loss

        inds = findall(rand(length(prob)) .≤ prob)
        arms = arms[inds]
        prms = prms[inds]

        n += 1
    end
    @debug "Finished SASHA"

    return first(prms)
end

sasha(T::Type, space::FiniteSpace, data::AbstractResampler; args::NamedTuple = (), temp::Real = 1, maximize::Bool = false) =
    sasha(T, collect(space), data, args = args, temp = temp, maximize = maximize)

end