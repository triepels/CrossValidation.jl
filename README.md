# Description
Provides simple and lightweight implementations of model validation and hyperparameter optimization for Julia. 

# Installation
Run the following code to install the package:
```
] add https://github.com/triepels/CrossValidation.jl
```

# Get Started
You have to define functions `fit!` and `loss` for your model type.

```julia
julia> import CrossValidation: fit!, loss
```

Function `fit!` takes a model and fits it on data based on some optional fitting arguments:

```julia
julia> function fit!(model::MyModel, data)
           # Code to fit model...
       end
```

Function `loss` estimates how well the model performs on (out-of-sample) data:

```julia
julia> function loss(model::MyModel, data)
           # Code to evalute loss of the model...
       end
```

# Features
Model validation based on various resample methods:
```julia
julia> validate(KFold(data, 10)) do train
            return fit!(MyModel(; args), train)
       end
```

Hyperparameter optimization using various optimizers:
```julia
julia> sp = space(a = DiscreteUniform(1:10))
julia> sha((x) -> MyModel(; x...), sp, FixedSplit(data, 0.8), Budget{:arg}(100))
```

Model validation with hyperparameter optimization:
```julia
julia> validate(KFold(data, 10)) do train
           args = brute((x) -> MyModel(; x...), sp, FixedSplit(train, 0.8))
           return fit!(MyModel(; args...), train)
       end
```

# Resample methods
The following resample methods are available:
* Fixed Split (Holdout)
* Random Split (Holdout)
* Leave-One-Out
* K-Fold
* Forward Chaining
* Sliding Window

# Optimizers
The following optimizers are available:
* Grid Search
* Random Search
* Hill-Climbing (HC)
* Successive Halving (SHA)
* Hyperband
* Simulated Annealing and Successive Halving (SASHA)

# Note
This package is not yet stable. Future releases might be subject to breaking changes.
