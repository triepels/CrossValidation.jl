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
julia> validate(MyModel(), KFold(data, 10))
```

Hyperparameter optimization using various optimizers:
```julia
julia> sp = space(a = DiscreteUniform(1:10))
julia> sha(MyModel, sp, FixedSplit(data, 0.8), Budget{:arg}(100))
```

Model validation with hyperparameter optimization:
```julia
julia> validate(KFold(data, 10)) do train
           parm = brute(MyModel, sp, FixedSplit(train, 0.8))
           return fit!(MyModel(; parm...), train)
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
