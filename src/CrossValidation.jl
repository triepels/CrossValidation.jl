module CrossValidation

using Base: @propagate_inbounds
using Random: shuffle!
using Distributed: @distributed, pmap

export DataSampler, FixedSplit, RandomSplit, KFold, ForwardChaining, SlidingWindow, PreProcess,
       ParameterSpace, ParameterSampler, GridSampler, RandomSampler,
       fit!, loss, validate, brute, hc, ConstantBudget, GeometricBudget, sha

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

function Base.iterate(r::PreProcess, state = 1)
    next = iterate(r.res, state)
    next === nothing && return nothing
    (train, test), state = next
    return r.f(train, test), state
end

struct ParameterSpace{names, T<:Tuple}
    args::T
end

function ParameterSpace(; args...)
    return ParameterSpace{keys(args), typeof(values(values(args)))}(values(values(args)))
end

Base.eltype(::Type{ParameterSpace{names, T}}) where {names, T} = NamedTuple{names, Tuple{map(eltype, T.parameters)...}}
Base.length(s::ParameterSpace) = length(s.args) == 0 ? 0 : prod(length, s.args)

Base.firstindex(s::ParameterSpace) = 1
Base.lastindex(s::ParameterSpace) = length(s)

Base.size(s::ParameterSpace) = length(s.args) == 0 ? (0,) : map(length, s.args)

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

_fit!(model, x::AbstractArray, args) = fit!(model, x; args...)
_fit!(model, x::Union{Tuple, NamedTuple}, args) = fit!(model, x...; args...)

fit!(model, x; args...) = throw(ErrorException("no fit! function defined for $(typeof(model))"))

_loss(model, x::AbstractArray) = loss(model, x)
_loss(model, x::Union{Tuple, NamedTuple}) = loss(model, x...)

loss(model, x...) = throw(ErrorException("no loss function defined for $(typeof(model))"))

@inline function _val(T, space, data, args)
    return sum(x -> _val_split(T, space, x..., args), data) / length(data)
end

function _val_split(T, space, train, test, args)
    models = pmap(x -> _fit!(T(; x...), train, args), space)
    loss = map(x -> _loss(x, test), models)
    @debug "Validated models" space=collect(space) args loss
    return loss
end

function validate(model, data::DataSampler; args...)
    @debug "Start model validation"
    loss = map(x -> _loss(_fit!(model, x[1], args), x[2]), data)
    @debug "Finished model validation"
    return loss
end

_f(f, x::AbstractArray) = f(x)
_f(f, x::Union{Tuple, NamedTuple}) = f(x...)

function validate(f::Function, data::DataSampler)
    @debug "Start model validation"
    loss = map(x -> _loss(_f(f, x[1]), x[2]), data)
    @debug "Finished model validation"
    return loss
end

function brute(T::Type, space::ParameterSampler, data::DataSampler, maximize::Bool = true; args...)
    length(space) ≥ 1 || throw(ArgumentError("nothing to optimize"))
    @debug "Start brute-force search"
    loss = _val(T, space, data, args)
    best = maximize ? argmax(loss) : argmin(loss)
    @debug "Finished brute-force search"
    return space[best]
end

function _candidates(space, blst, state, k)
    dim = size(space)
    cands = sizehint!(Int[], 2 * k * length(dim))
    @inbounds for i in eachindex(dim)
        if i == 1
            ind = mod(state - 1, dim[1]) + 1
            for j in reverse(1:k)
                if ind - j ≥ 1
                    cand = state - j
                    if cand ∉ blst
                        push!(cands, cand)
                    end
                end
            end
            for j in 1:k
                if ind + j ≤ dim[1]
                    cand = state + j
                    if cand ∉ blst
                        push!(cands, cand)
                    end
                end
            end
        else
            ind = mod((state - 1) ÷ dim[i - 1], dim[i]) + 1
            for j in reverse(1:k)
                if ind - j ≥ 1
                    cand = state - j * dim[i - 1]
                    if cand ∉ blst
                        push!(cands, cand)
                    end
                end
            end
            for j in 1:k
                if ind + j ≤ dim[i]
                    cand = state + j * dim[i - 1]
                    if cand ∉ blst
                        push!(cands, cand)
                    end
                end
            end
        end
    end
    return cands
end

function hc(T::Type, space::ParameterSpace, data::DataSampler, k::Int = 1, maximize::Bool = true; args...)
    m = length(space)
    m ≥ 1 || throw(ArgumentError("nothing to optimize"))
    k ≥ 1 || throw(ArgumentError("cannot generate $k candidates in each direction"))

    best, loss = rand(1:m), maximize ? -Inf : Inf
    cands, blst = [best], Int[]

    @debug "Start hill-climbing"
    while !isempty(cands)
        append!(blst, cands)
        curr = _val(T, space[cands], data, args)
        if maximize
            ind = argmax(curr)
            if loss ≥ curr[ind] 
                break
            end
        else
            ind = argmin(curr)
            if loss ≤ curr[ind]
                break
            end
        end
        best, loss = cands[ind], curr[ind]
        cands = _candidates(space, blst, best, k)
    end
    @debug "Finished hill-climbing"

    return space[best]
end

abstract type Budget end

_cast(T::Type{A}, x::Integer) where A <: AbstractFloat = T(x)
_cast(T::Type{A}, x::AbstractFloat) where A <: Integer = floor(T, x)
_cast(T::Type{A}, x::Number) where A <: Number = x

struct ConstantBudget{names, T<:Tuple{Vararg{Number}}} <: Budget
    args::NamedTuple{names, T}
end

getbudget(b::ConstantBudget) = b.args

function getbudget(b::ConstantBudget, i::Int, n::Int)
    return map(x -> _cast(typeof(x), x / n), b.args)
end

struct GeometricBudget{names, T<:Tuple{Vararg{Number}}} <: Budget
    args::NamedTuple{names, T}
    rate::Number
end

function GeometricBudget(args::NamedTuple{names, T}; rate::Number = 2) where {names, T<:Tuple{Vararg{Number}}}
    return GeometricBudget(args, rate)
end

getbudget(b::GeometricBudget) = b.args

function getbudget(b::GeometricBudget, i::Int, n::Int)
    r = b.rate
    return map(x -> _cast(typeof(x), r^(i - 1) * x * (r - 1) / (r^n - 1)), b.args)
end

_halve!(x::Vector) = resize!(x, ceil(Int, length(x) / 2))

function sha(T::Type, space::ParameterSampler, data::DataSampler, budget::Budget, maximize::Bool = true)
    m, n = length(space), length(data)
    m ≥ 1 || throw(ArgumentError("nothing to optimize"))
    n == 1 || throw(ArgumentError("cannot optimize by $n resample folds"))
    
    train, test = first(data)
    arms = map(x -> T(; x...), space)
    prms = collect(space)

    k = floor(Int, log2(m))
    @debug "Start successive halving"
    for i in 1:k
        args = getbudget(budget, i, k)
        arms = pmap(x -> _fit!(x, train, args), arms)
        loss = map(x -> _loss(x, test), arms)
        @debug "Validated arms" space=prms args loss
        inds = sortperm(loss, rev=maximize)
        arms = _halve!(arms[inds])
        prms = _halve!(prms[inds])
    end
    @debug "Finished successive halving"

    return prms[1]
end

end
