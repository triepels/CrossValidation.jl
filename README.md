# Description
Provides simple and lightweight implementations of model validation and hyperparameter optimization for Julia. 

# Installation
Run the following code to install the package:
```julia
] add https://github.com/triepels/CrossValidation.jl
```

# Get Started
You have to define functions `fit!` and `loss` for your model type.

```julia
julia> import CrossValidation: fit!, loss
```

Function `fit!` takes a model and fits it on data based on some optional fitting arguments:

```julia
julia> function fit!(model::MyModel, data; args)
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
julia> validate(MyModel(parms), KFold(data), args)
```

Hyperparameter optimization using various optimizers:
```julia
julia> sp = space(a = -6.0:0.5:6.0, b = -6.0:0.5:6.0)
julia> sha(MyModel, sp, FixedSplit(data), GeometricBudget(args), 0.5, false)
```

Model validation with hyperparameter optimization:
```julia
julia> validate(KFold(data)) do train
           parms = brute(MyModel, sp, FixedSplit(train), false, args)
           return fit!(MyModel(parms...), train)
       end
```

# Resample methods
The following resample methods are available:
* Fixed Split
* Random Split (holdout)
* Leave-One-Out
* K-fold
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
