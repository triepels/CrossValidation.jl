module CrossValidation

using Base: @propagate_inbounds, OneTo
using Random: GLOBAL_RNG, AbstractRNG, shuffle!
using Distributed: @distributed, pmap

import Random: rand

export DataSampler, FixedSplit, RandomSplit, LeaveOneOut, KFold, ForwardChaining, SlidingWindow,
       AbstractSpace, FiniteSpace, InfiniteSpace, space, ParameterVector,
       AbstractDistribution, DiscreteDistribution, ContinousDistribution, DiscreteUniform, Uniform, LogUniform, Normal, sample,
       AbstractBudget, AbstractRoundBudget, AbstractOverallBudget, schedule,
       fit!, loss, validate, brute, hc, ConstantBudget, GeometricBudget, HyperBudget, sha, hyperband, sasha

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

getobs(x, i) = x[i]
getobs(x::AbstractArray, i) = x[Base.setindex(map(Base.Slice, axes(x)), i, ndims(x))...]
getobs(x::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(getobs, i), x)

restype(x) = eltype(x)
restype(x::Tuple) = Tuple{map(restype, x)...}
restype(x::NamedTuple) = NamedTuple{keys(x), Tuple{map(restype, x)...}}
restype(x::AbstractArray) = Array{eltype(x), ndims(x)}

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

FixedSplit(data, ratio::Number = 0.8) = FixedSplit(data, floor(Int, nobs(data) * ratio))

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

RandomSplit(data, ratio::Number = 0.8) = RandomSplit(data, floor(Int, nobs(data) * ratio))

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
    function KFold(data; k::Int = 10)
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
    if n == m
        return shuffle!(collect(iter))
    end
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

struct DiscreteUniform{T} <: DiscreteDistribution
    states::T
end

Base.eltype(::Type{DiscreteUniform{T}}) where T = eltype(T)
Base.length(d::DiscreteUniform) = length(d.states)
Base.getindex(d::DiscreteUniform, i::Int) = getindex(d.states, i)

rand(rng::AbstractRNG, d::DiscreteUniform) = rand(rng, d.states)

struct Uniform{T<:AbstractFloat} <: ContinousDistribution
    a::T
    b::T
    function Uniform(a::T, b::T) where T<:AbstractFloat
        a < b || throw(ArgumentError("a must be smaller than b"))
        return new{T}(a, b)
    end
end

Uniform(a::AbstractFloat, b::AbstractFloat) = Uniform(promote(a, b)...)

rand(rng::AbstractRNG, d::Uniform{T}) where T = d.a + (d.b - d.a) * rand(rng, T)

struct LogUniform{T<:AbstractFloat} <: ContinousDistribution
    a::T
    b::T
    function LogUniform(a::T, b::T) where T<:AbstractFloat
        a < b || throw(ArgumentError("a must be smaller than b"))
        return new{T}(a, b)
    end
end

LogUniform(a::AbstractFloat, b::AbstractFloat) = LogUniform(promote(a, b)...)

rand(rng::AbstractRNG, d::LogUniform{T}) where T = exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng, T))

struct Normal{T<:AbstractFloat} <: ContinousDistribution
    mean::T
    std::T
    function Normal(mean::T, std::T) where T<:AbstractFloat
        std > zero(std) || throw(ArgumentError("standard deviation must be larger than zero"))
        return new{T}(mean, std)
    end
end

Normal(mean::AbstractFloat, std::AbstractFloat) = Normal(promote(mean, std)...)

rand(rng::AbstractRNG, d::Normal{T}) where T = d.mean + d.std * randn(rng, T)

abstract type AbstractSpace end

struct FiniteSpace{names, T<:Tuple} <: AbstractSpace
    vars::T
end

Base.eltype(::Type{FiniteSpace{names, T}}) where {names, T} = NamedTuple{names, Tuple{map(eltype, T.parameters)...}}
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

function validate(model, data::AbstractResampler; args...)
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

function brute(T::Type, prms::ParameterVector, data::AbstractResampler, maximize::Bool = true; args...)
    length(prms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    @debug "Start brute-force search"
    loss = _val(T, prms, data, values(args))
    ind = maximize ? argmax(loss) : argmin(loss)
    @debug "Finished brute-force search"
    return prms[ind]
end

brute(T::Type, space::FiniteSpace, data::AbstractResampler, maximize::Bool = true; args...) =
    brute(T, collect(space), data, maximize; args...)

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

function hc(T::Type, space::FiniteSpace, data::AbstractResampler, nstart::Int = 1, k::Int = 1, maximize::Bool = true; args...)
    length(space) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    k ≥ 1 || throw(ArgumentError("invalid neighborhood size of $k"))

    bl = Int[]
    parm = nothing
    best = maximize ? -Inf : Inf

    cand = sample(OneTo(length(space)), nstart)

    @debug "Start hill-climbing"
    while !isempty(cand)
        append!(bl, cand)

        loss = _val(T, space[cand], data, values(args))
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

abstract type AbstractBudget{names, T<:Tuple{Vararg{Number}}} end
abstract type AbstractHyperBudget{names, T<:Tuple{Number}} <: AbstractBudget{names, T} end

_cast(T::Type{A}, x::Integer, r::RoundingMode) where A <: AbstractFloat = T(x)
_cast(T::Type{A}, x::AbstractFloat, r::RoundingMode) where A <: Integer = round(T, x, r)
_cast(T::Type{A}, x::Number, r::RoundingMode) where A <: Number = x

struct GeometricBudget{names, T} <: AbstractBudget{names, T}
    args::T
end

function GeometricBudget(; args...)
    return GeometricBudget{keys(args), typeof(values(values(args)))}(values(values(args)))
end

Base.values(budget::GeometricBudget) = budget.args

function schedule(budget::GeometricBudget{names, T}, n, rate) where {names, T}
    b = floor(Int, log(rate, n)) + 1
    return schedule(budget, b, n, rate)
end

function schedule(budget::GeometricBudget{names, T}, b, n, rate) where {names, T}
    k = Vector{Int}(undef, b)
    args = Vector{NamedTuple{names, T}}(undef, b)
    for i in OneTo(b)
        c = ceil(Int, n / rate^(i - 1))
        args[i] = NamedTuple{names, T}(map(x -> _cast(typeof(x), x / (c * b), RoundDown), budget.args))
        k[i] = ceil(Int, c / rate)
    end
    return k, args
end

struct ConstantBudget{names, T} <: AbstractBudget{names, T}
    args::T
end

function ConstantBudget(; args...)
    return ConstantBudget{keys(args), typeof(values(values(args)))}(values(values(args)))
end

Base.values(budget::ConstantBudget) = budget.args

function schedule(budget::ConstantBudget{names, T}, n, rate) where {names, T}
    b = floor(Int, log(rate, n)) + 1
    return schedule(budget, b, n, rate)
end

function schedule(budget::ConstantBudget{names, T}, b, n, rate) where {names, T}
    k = Vector{Int}(undef, b)
    args = Vector{NamedTuple{names, T}}(undef, b)
    c = (rate - 1) * rate^(b - 1) / (n * (rate^b - 1))
    for i in OneTo(b)
        args[i] = NamedTuple{names, T}(map(x -> _cast(typeof(x), c * x, RoundDown), budget.args))
        k[i] = ceil(Int, n / rate^i)
    end
    return k, args
end

struct HyperBudget{names, T} <: AbstractHyperBudget{names, T}
    args::T
end

function HyperBudget(; args...)
    return HyperBudget{keys(args), typeof(values(values(args)))}(values(values(args)))
end

Base.values(budget::HyperBudget) = budget.args

function schedule(budget::HyperBudget{names, T}, n, rate) where {names, T}
    b = floor(Int, log(rate, first(budget.args))) + 1
    return schedule(budget, b, n, rate)
end

function schedule(budget::HyperBudget{names, T}, b, n, rate) where {names, T}
    k = Vector{Int}(undef, b)
    args = Vector{NamedTuple{names, T}}(undef, b)
    for i in OneTo(b)
        c = rate^(i - b)
        args[i] = NamedTuple{names, T}(_cast(typeof(first(budget.args)), c * first(budget.args), RoundNearest)) #RoundDown?
        k[i] = max(floor(Int, n / rate^i), 1)
    end
    return k, args
end

function sha(T::Type, prms::ParameterVector, data::AbstractResampler, budget::AbstractBudget, rate::Number, maximize::Bool = true)
    length(prms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    length(data) == 1 || throw(ArgumentError("can only optimize over one resample fold"))
    rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))

    train, test = first(data)
    arms = map(x -> T(; x...), prms)

    bracket = schedule(budget, length(arms), float(rate))

    @debug "Start successive halving"
    for (k, args) in zip(bracket...)
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

sha(T::Type, space::FiniteSpace, data::AbstractResampler, budget::AbstractBudget, rate::Number, maximize::Bool = true) =
    sha(T, collect(space), data, budget, rate, maximize)

function hyperband(T::Type, space::AbstractSpace, data::AbstractResampler, budget::AbstractHyperBudget, rate::Number, maximize::Bool = true)
    length(data) == 1 || throw(ArgumentError("can only optimize over one resample fold"))
    rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))

    parm = nothing
    best = maximize ? -Inf : Inf

    train, test = first(data)
    m = floor(Int, log(rate, first(values(budget)))) + 1

    @debug "Start hyperband"
    for b in reverse(OneTo(m))
        n = ceil(Int, m * rate^(b - 1) / b)

        loss = nothing
        prms = sample(space, n)
        arms = map(x -> T(; x...), prms)
        
        bracket = schedule(budget, b, n, float(rate))

        @debug "Start successive halving"
        for (k, args) in zip(bracket...)
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

function sasha(T::Type, prms::ParameterVector, data::AbstractResampler, temp::Number, maximize::Bool = true; args...)
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

sasha(T::Type, space::FiniteSpace, data::AbstractResampler, temp::Number, maximize::Bool = true; args...) =
    sasha(T, collect(space), data, temp, maximize; args...)

end
