# Description
Provides simple and lightweight implementations of model validation and hyperparameter search for Julia. 

# Features
Validation schemes:
* Hyperparameter search
* Model validation
* Nested cross-validation

Resample methods:
* Holdout
* LeavePOut
* KFold

Search methods:
* Exhaustive search

# Installation

```julia
] add https://github.com/triepels/CrossValidation.jl
```

# Examples
The examples below demonstrate some common use cases of the `CrossValidation.jl` package.

## Supervised Learning Model
Suppose we have a classifier of type `MyClassifier` which has two hyperparameters, `a` and `b`. The model is defined by the following struct:

```julia
struct MyClassifier
    a::Int
    b::Int
end
```

We define the following function to train the classifier on some training data:

```julia
function myclassifier(x::AbstractArray, y::AbstractArray, a::Int, b::Int)
    # Code to fit the classifier...
end
```

The `CrossValidation.jl` package works based on two functions, `predict` and `score`, which the user needs to override. The `predict` function generates predictions by a fitted model on some data.

```julia
function predict(model::MyClassifier, x::AbstractArray)
    # Code to generate predictions...
end
```

The `score` function 'scores' how well a model performs on some data. In the case of supervised learning, this will be classification accuracy or a similar score.

```julia
function score(model::MyClassifier, x::AbstractArray, y::AbstractArray)
    # Code to score the performance of the model...
end
```

## Hyperparameter Search
The hyperparameters `a` and `b` of `MyClassifier` cannot be deduced directly from data. Instead, these parameters need to be tuned by a hyperparameter search. The `CrossValidation.jl` package allows doing this based on an exhaustive search in combination with various resampling methods. Let us define the following grid:

```julia
search = ExhaustiveSearch(a=1:2, b=3:4)
```

Accordingly, we need to specify the resampling method and corresponding data that is used for the search. We will use k-fold cross-validation with five folds:

```julia
method = KFold((x, y), k=5)
```

Here, `x` is an array with training data and `y` an array with the corresponding training labels. Finally, we start the hyperparameter tuning by:

```julia
cv = crossvalidate(mymodel, search, method)
```

The result of the search is an object of type `ParameterSearch` with three data members: 
* `models`: an array of models that are fitted during the parameter search.
* `scores`: an array holding the scores of the models as estimated by the `score` function. The array has a size of `n` x `m`, where `n` is the number of folds and `m` the number of parameter configurations.
* `final`: the final model that is fitted on the entire dataset based on the parameter configuration that performed on average the best over all folds.

We can use function `predict` to generate predictions by the final model:

```julia
predict(cv, x)
```

Similarly, we can score the final model on some data by the `score` function:

```julia
score(cv, x, y)
```

## Unsupervised Learning Model
The `CrossValidation.jl` package can also be used in an unsupervised learning setting. Suppose we have an unsupervised learning model of type `MyModel`. The model is defined by the struct:

```julia
struct MyModel
    a::Int
    b::Int
end
```

We define the following function to fit the model on some training data: 

```julia
function mymodel(x::AbstractArray, a::Int, b::Int)
    # Code to fit model here...
end
```

We also override the `predict` and `score` functions.

```julia
function predict(model::MyModel, x::AbstractArray)
    # Code to generate predictions here...
end
```
```julia
function score(model::MyModel, x::AbstractArray)
    # Code to score the performance of a model here...
end
```

We can now use the `CrossValidation.jl` package similarly as in the supervised learning example. We will demonstrate this by performing model validation.

## Model Validation
It might be of interest to validate how well `MyModel` performs out of sample. We do this by performing model validation. The `CrossValidation.jl` allows performing model validation based on various resampling methods. Let us use leave-p-out with `p` set to 100:

```julia
method = LeavePOut(x, p=100)
```

We start the model validation by:

```julia
crossvalidate(mymodel, method)
```

The result of the validation is an object of type `ModelValidation` with two data members:
* `models`: an array of models that are fitted during the model validation.
* `scores`: an array holding the scores of the models as estimated by the `score` function. The array has a size of `n` x 1, where `n` is the number of folds.

## Nested Cross-Validation
In some cases, a model needs to be validated on some data, but the model has some hyperparameters that need to be tuned for each fold as well. This can be done by nested cross-validation. Performing nested cross-validation using the `CrossValidation.jl` package is straightforward. Let us perform nested cross-validation on `MyClassifier` as defined in the previous examples. First, we define a grid of parameters that we want to try out in the inner loop:

```julia
search = ExhaustiveSearch(a=1:2, b=3:4)
```

Accordingly, we start the nested cross-validation by:

```julia
crossvalidate(crossvalidate(myclassifier, HoldOut((x, y)), search), KFold((x, y)))
```

Here, we use holdout validation for the inner loop and k-fold cross-validation for the outer loop. The result of the nested cross-validation is an object of type `ModelValidation` with two data members:
* `models`: an array of `ParameterSearch` objects that are created by the inner loop of the nested cross-validation.
* `scores`: an array holding the scores of the best performing models of each parameter search as estimated by the `score` function. The array has a size of `n` x 1, where `n` is the number of folds.