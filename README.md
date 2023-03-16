# Description
Provides simple and lightweight implementations of model validation and hyperparameter search for Julia. 

# Features
Model validation based on various resample methods.
```julia
julia> validate(MyModel(2.0, 2.0), KFold(x), epochs=100)
```

Hyperparameter optimization using various optimizers.
```julia
julia> space = Space(a = -6.0:0.5:6.0, b = -6.0:0.5:6.0)
julia> sha(MyModel, space, FixedSplit(x), GeometricBudget(epochs=100), 0.5, false)
```

Model validation with hyperparameter optimization:
```julia
julia> validate(KFold(x)) do train
           parms = brute(MyModel, space, FixedSplit(train), false, epochs=100)
           return fit!(MyModel(parms...), train)
       end
```

# Resample methods
The following resample methods are available:
* Fixed split
* Random split (holdout)
* K-fold
* Forward chaining
* Sliding window

# Optimizers
The following optimizers are available:
* Grid search
* Random search
* Hill-Climbing (HC)
* Successive Halving (SHA)
* Simulated Annealing and Successive Halving (SASHA)

# Installation

```julia
] add https://github.com/triepels/CrossValidation.jl
```

# Note
This package is not yet stable. Future releases might be subject to breaking changes.
