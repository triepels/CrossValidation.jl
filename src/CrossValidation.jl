module CrossValidation

using Base: @propagate_inbounds, OneTo
using Random: GLOBAL_RNG, AbstractRNG, shuffle!
using Distributed: @distributed, pmap

import Random: rand

export DataSampler, FixedSplit, RandomSplit, LeaveOneOut, KFold, ForwardChaining, SlidingWindow,
       AbstractSpace, FiniteSpace, InfiniteSpace, ParameterVector,
       AbstractBudget, AbstractRoundBudget, AbstractOverallBudget, schedule,
       AbstractDistribution, Uniform, Normal, sample,
       fit!, loss, validate, brute, hc, ConstantBudget, GeometricBudget, HyperBudget, sha, hyperband, sasha

nobs(x::Any) = 1
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

abstract type AbstractResampler end

Base.eltype(r::AbstractResampler) = Tuple{restype(r.data), restype(r.data)}

struct FixedSplit{D} <: AbstractResampler
    data::D
    nobs::Int
    m::Int
end

function FixedSplit(data::Union{AbstractArray, Tuple, NamedTuple}, ratio::Number = 0.8)
    n = nobs(data)
    0 < n * ratio < n || throw(ArgumentError("data cannot be split based on a $ratio ratio"))
    return FixedSplit(data, n, ceil(Int, n * ratio))
end

function FixedSplit(data::Union{AbstractArray, Tuple, NamedTuple}, m::Int)
    n = nobs(data)
    0 < m < n || throw(ArgumentError("data cannot be split by $m"))
    return FixedSplit(data, n, m)
end

Base.length(r::FixedSplit) = 1

@propagate_inbounds function Base.iterate(r::FixedSplit, state = 1)
    state > 1 && return nothing
    train = getobs(r.data, OneTo(r.m))
    test = getobs(r.data, (r.m + 1):r.nobs)
    return (train, test), state + 1
end

struct RandomSplit{D} <: AbstractResampler
    data::D
    nobs::Int
    m::Int
    perm::Vector{Int}
end

function RandomSplit(data::Union{AbstractArray, Tuple, NamedTuple}, ratio::Number = 0.8)
    n = nobs(data)
    0 < ratio * n < n || throw(ArgumentError("data cannot be split based on a $ratio ratio"))
    return RandomSplit(data, n, ceil(Int, ratio * n), shuffle!([OneTo(n);]))
end

function RandomSplit(data::Union{AbstractArray, Tuple, NamedTuple}, m::Int)
    n = nobs(data)
    0 < m < n || throw(ArgumentError("data cannot be split by $m"))
    return RandomSplit(data, n, m, shuffle!([OneTo(n);]))
end

Base.length(r::RandomSplit) = 1

@propagate_inbounds function Base.iterate(r::RandomSplit, state = 1)
    state > 1 && return nothing
    train = getobs(r.data, r.perm[OneTo(r.m)])
    test = getobs(r.data, r.perm[(r.m + 1):r.nobs])
    return (train, test), state + 1
end

struct LeaveOneOut{D} <: AbstractResampler
    data::D
    nobs::Int
end

function LeaveOneOut(data::Union{AbstractArray, Tuple, NamedTuple})
    n = nobs(data)
    n > 1 || throw(ArgumentError("data has too few observations to split"))
    return LeaveOneOut(data, n)
end

Base.length(r::LeaveOneOut) = r.nobs

@propagate_inbounds function Base.iterate(r::LeaveOneOut, state = 1)
    state > length(r) && return nothing
    train = getobs(r.data, union(OneTo(state - 1), (state + 1):r.nobs))
    test = getobs(r.data, state:state)
    return (train, test), state + 1
end

struct KFold{D} <: AbstractResampler
    data::D
    nobs::Int
    k::Int
    perm::Vector{Int}
end

function KFold(data::Union{AbstractArray, Tuple, NamedTuple}; k::Int = 10)
    n = nobs(data)
    1 < k ≤ n || throw(ArgumentError("data cannot be partitioned into $k folds"))
    return KFold(data, n, k, shuffle!([OneTo(n);]))
end

Base.length(r::KFold) = r.k

@propagate_inbounds function Base.iterate(r::KFold, state = 1)
    state > length(r) && return nothing
    m = mod(r.nobs, r.k)
    w = floor(Int, r.nobs / r.k)
    fold = ((state - 1) * w + min(m, state - 1) + 1):(state * w + min(m, state))
    train = getobs(r.data, r.perm[setdiff(OneTo(r.nobs), fold)])
    test = getobs(r.data, r.perm[fold])
    return (train, test), state + 1
end

struct ForwardChaining{D} <: AbstractResampler
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
    train = getobs(r.data, OneTo(r.init + (state - 1) * r.out))
    test = getobs(r.data, (r.init + (state - 1) * r.out + 1):min(r.init + state * r.out, r.nobs))
    return (train, test), state + 1
end

struct SlidingWindow{D} <: AbstractResampler
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

abstract type AbstractSpace end

struct FiniteSpace{names, T<:Tuple} <: AbstractSpace
    vars::T
end

function FiniteSpace(; vars...)
    return FiniteSpace{keys(vars), typeof(values(values(vars)))}(values(values(vars)))
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
sample(space::FiniteSpace) = sample(GLOBAL_RNG, space)

function _sample(rng, iter, n)
    m = length(iter)
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

_sample(iter, n) = _sample(GLOBAL_RNG, iter, n)

function sample(rng::AbstractRNG, space::FiniteSpace, n::Int)
    m = length(space)
    1 ≤ n ≤ m || throw(ArgumentError("cannot sample $n times without replacement from space"))
    return [space[i] for i in _sample(rng, OneTo(length(space)), n)]
end

sample(space::FiniteSpace, n::Int) = sample(GLOBAL_RNG, space, n)

abstract type AbstractDistribution end

struct Uniform{T<:AbstractFloat} <: AbstractDistribution
    a::T
    b::T
end

rand(rng::AbstractRNG, d::Uniform{T}) where T = d.a + (d.b - d.a) * rand(rng, T)

struct Normal{T<:AbstractFloat} <: AbstractDistribution
    mean::T
    std::T
end

rand(rng::AbstractRNG, d::Normal{T}) where T = d.mean + d.std * randn(rng, T)

struct InfiniteSpace{names, T<:Tuple} <: AbstractSpace
    vars::T
end

function InfiniteSpace(; vars...)
    return InfiniteSpace{keys(vars), typeof(values(values(vars)))}(values(values(vars)))
end

rand(rng::AbstractRNG, space::InfiniteSpace{names}) where {names} = NamedTuple{names}(map(x -> rand(rng, x), space.vars))

sample(rng::AbstractRNG, space::InfiniteSpace) = rand(rng, space)
sample(space::InfiniteSpace) = sample(GLOBAL_RNG, space)

sample(rng::AbstractRNG, space::InfiniteSpace, n::Int) = [sample(rng, space) for _ in OneTo(n)]
sample(space::InfiniteSpace, n::Int) = sample(GLOBAL_RNG, space, n)

const ParameterVector = Array{NamedTuple{names, T}, 1} where {names, T}

_fit!(model, x::AbstractArray, args) = fit!(model, x; args...)
_fit!(model, x::Union{Tuple, NamedTuple}, args) = fit!(model, x...; args...)

fit!(model, x; args...) = throw(ErrorException("no fit! function defined for $(typeof(model))"))

_loss(model, x::AbstractArray) = loss(model, x)
_loss(model, x::Union{Tuple, NamedTuple}) = loss(model, x...)

loss(model, x...) = throw(ErrorException("no loss function defined for $(typeof(model))"))

@inline function _val(T, prms, data, args)
    return sum(x -> _val_split(T, prms, x..., args), data) / length(data)
end

function _val_split(T, prms, train, test, args)
    models = pmap(x -> _fit!(T(; x...), train, args), prms)
    loss = map(x -> _loss(x, test), models)
    @debug "Validated models" prms args loss
    return loss
end

function validate(model::Any, data::AbstractResampler; args...)
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

    cand = _sample(OneTo(length(space)), nstart)

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

abstract type AbstractBudget end

abstract type AbstractRoundBudget <: AbstractBudget end
abstract type AbstractOverallBudget <: AbstractBudget end

_cast(T::Type{A}, x::Integer, r::RoundingMode) where A <: AbstractFloat = T(x)
_cast(T::Type{A}, x::AbstractFloat, r::RoundingMode) where A <: Integer = round(T, x, r)
_cast(T::Type{A}, x::Number, r::RoundingMode) where A <: Number = x

struct GeometricBudget{names, T<:Tuple{Vararg{Number}}} <: AbstractOverallBudget
    args::T
end

function GeometricBudget(; args...)
    return GeometricBudget{keys(args), typeof(values(values(args)))}(values(values(args)))
end

function schedule(budget::GeometricBudget{names, T}, n, rate) where {names, T}
    b = floor(Int, log(rate, n)) + 1
    return schedule(budget, b, n, rate)
end

function schedule(budget::GeometricBudget{names, T}, b, n, rate) where {names, T}
    brkt = Vector{Tuple{Int, NamedTuple{names, T}}}(undef, b)
    for i in OneTo(b)
        k = ceil(Int, n / rate^(i - 1))
        args = NamedTuple{names, T}(map(x -> _cast(typeof(x), x / (k * b), RoundDown), budget.args))
        brkt[i] = (ceil(Int, k / rate), args)
    end
    return brkt
end

struct ConstantBudget{names, T<:Tuple{Vararg{Number}}} <: AbstractOverallBudget
    args::T
end

function ConstantBudget(; args...)
    return ConstantBudget{keys(args), typeof(values(values(args)))}(values(values(args)))
end

function schedule(budget::ConstantBudget{names, T}, n, rate) where {names, T}
    b = floor(Int, log(rate, n)) + 1
    return schedule(budget, b, n, rate)
end

function schedule(budget::ConstantBudget{names, T}, b, n, rate) where {names, T}
    c = (rate - 1) * rate^(b - 1) / (n * (rate^b - 1))
    brkt = Vector{Tuple{Int, NamedTuple{names, T}}}(undef, b)
    for i in OneTo(b)
        args = NamedTuple{names, T}(map(x -> _cast(typeof(x), c * x, RoundDown), budget.args))
        brkt[i] = (ceil(Int, n / rate^i), args)
    end
    return brkt
end

struct HyperBudget{names, T<:NTuple{1, Number}} <: AbstractRoundBudget
    args::T
end

function HyperBudget(; args...)
    return HyperBudget{keys(args), typeof(values(values(args)))}(values(values(args)))
end

function schedule(budget::HyperBudget{names, T}, n, rate) where {names, T}
    b = floor(Int, log(rate, budget.args[1])) + 1
    return schedule(budget, b, n, rate)
end

function schedule(budget::HyperBudget{names, T}, b, n, rate) where {names, T}
    brkt = Vector{Tuple{Int, NamedTuple{names, T}}}(undef, b)
    for i in OneTo(b)
        c = rate^(i - b)
        args = NamedTuple{names, T}(map(x -> _cast(typeof(x), c * x, RoundNearest), budget.args))
        brkt[i] = (ceil(Int, n / rate^i), args)
    end
    return brkt
end

function sha(T::Type, prms::ParameterVector, data::AbstractResampler, budget::AbstractBudget, rate::Number, maximize::Bool = true)
    length(prms) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    length(data) == 1 || throw(ArgumentError("can only optimize over one resample fold"))
    rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))

    train, test = first(data)
    arms = map(x -> T(; x...), prms)

    @debug "Start successive halving"
    for (k, args) in schedule(budget, length(arms), float(rate))
        arms = pmap(x -> _fit!(x, train, args), arms)
        loss = map(x -> _loss(x, test), arms)
        @debug "Validated arms" prms args loss

        inds = sortperm(loss, rev=maximize)
        arms = arms[inds[OneTo(k)]]
        prms = prms[inds[OneTo(k)]]
    end
    @debug "Finished successive halving"

    return prms[1]
end

sha(T::Type, space::FiniteSpace, data::AbstractResampler, budget::AbstractBudget, rate::Number, maximize::Bool = true) =
    sha(T, collect(space), data, budget, rate, maximize)

function hyperband(T::Type, space::AbstractSpace, data::AbstractResampler, budget::AbstractRoundBudget, rate::Number, maximize::Bool = true)
    length(data) == 1 || throw(ArgumentError("can only optimize over one resample fold"))
    rate > 1 || throw(ArgumentError("unable to discard arms with rate $rate"))

    parm = nothing
    best = maximize ? -Inf : Inf

    train, test = first(data)
    m = floor(Int, log(rate, budget.args[1])) + 1

    @debug "Start hyperband"
    for b in reverse(OneTo(m))
        n = ceil(Int, m * rate^(b - 1) / b)

        loss = nothing
        prms = sample(space, n)
        arms = map(x -> T(; x...), prms)
        
        @debug "Start successive halving"
        for (k, args) in schedule(budget, b, n, float(rate))
            arms = pmap(x -> _fit!(x, train, args), arms)
            loss = map(x -> _loss(x, test), arms)
            @debug "Validated arms" prms args loss
    
            inds = sortperm(loss, rev=maximize)
            arms = arms[inds[OneTo(k)]]
            prms = prms[inds[OneTo(k)]]    
        end

        if maximize
            if loss[1] > best
                parm = prms[1]
                best = loss[1]
            end
        else
            if loss[1] < best
                parm = prms[1]
                best = loss[1]
            end
        end
        @debug "Finished successive halving"
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
    while length(arms) > 1
        arms = pmap(x -> _fit!(x, train, args), arms)
        loss = map(x -> _loss(x, test), arms)
        @debug "Validated arms" prms prob loss

        if maximize
            prob = exp.(n .* (loss .- max(loss...)) ./ temp)
        else
            prob = exp.(-n .* (loss .- min(loss...)) ./ temp)
        end        

        inds = findall(rand(length(prob)) .≤ prob)
        arms = arms[inds]
        prms = prms[inds]

        n += 1
    end

    budget = map(x -> n * x, values(args))

    return prms[1], budget
end

sasha(T::Type, space::FiniteSpace, data::AbstractResampler, temp::Number, maximize::Bool = true; args...) =
    sasha(T, collect(space), data, temp, maximize; args...)

end
