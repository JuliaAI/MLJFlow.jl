# MLJFlow.jl

| Branch | Build | Coverage |
| :---: | :---: | :---: |
| dev | [![Continuous Integration (CPU)][ci-dev-img]][ci-dev] | [![Code Coverage][codecov-dev-img]][codecov-dev] |

[ci-dev]: https://github.com/pebeto/MLJFlow.jl/actions/workflows/CI.yml
[ci-dev-img]: https://github.com/pebeto/MLJFlow.jl/actions/workflows/CI.yml/badge.svg?branch=dev "Continuous Integration (CPU)"
[codecov-dev]: https://codecov.io/github/JuliaAI/MLJFlow.jl?branch=dev
[codecov-dev-img]: https://codecov.io/gh/JuliaAI/MLJFlow.jl/branch/dev/graphs/badge.svg?branch=dev "Code Coverage"

[MLJ](https://github.com/alan-turing-institute/MLJ.jl) is a Julia framework for
combining and tuning machine learning models. MLJFlow is a package that extends
the MLJ capabilities to use [MLflow](https://mlflow.org/) as a backend for
model tracking and experiment management. To be specific, MLJFlow provides a
close to zero-preparation to use MLflow with MLJ; by the usage of function
extensions that automate the MLflow cycle (create experiment, create run, log
metrics, log parameters, log artifacts, etc.).

## Background

This project is part of the GSoC 2023 program. The proposal description can be
found [here](https://summerofcode.withgoogle.com/programs/2023/projects/iRxuzeGJ).
The entire workload is divided into three different repositories:
[MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl), 
[MLFlowClient.jl](https://github.com/JuliaAI/MLFlowClient.jl) and this one.

## Features

- [x] MLflow cycle automation (create experiment, create run, log metrics, log parameters,
      log artifacts, etc.)

- [x] Provides a wrapper `Logger` for MLFlowClient.jl clients and associated
      metadata; instances of this type are valid "loggers", which can be passed to MLJ
      functions supporting the `logger` keyword argument.
	  
- [x] Provides MLflow integration with MLJ's `evaluate!`/`evaluate` method (model
      **performance evaluation**)

- [x] Extends MLJ's `MLJ.save` method, to save trained machines as retrievable MLflow
      client artifacts

- [ ] Provides MLflow integration with MLJ's `TunedModel` wrapper (to log **hyper-parameter
      tuning** workflows)

- [ ] Provides MLflow integration with MLJ's `IteratedModel` wrapper (to log **controlled
      iteration** of tree gradient boosters, neural networks, and other iterative models)

- [x] Plays well with **composite models** (pipelines, stacks, etc.)


## Examples

### Logging a model performance evaluation

The example below assumes the user is familiar with basic MLflow concepts. We suppose an
MLflow API service is running on a local server, with address "http://127.0.0.1:5000". (In a
shell/console, run `mlflow server` to launch an mlflow service on a local server.)

Refer to the [MLflow documentation](https://www.mlflow.org/docs/latest/index.html) for
necessary background.

We assume MLJDecisionTreeClassifier is in the user's active Julia package
environment.

```julia
using MLJ # Requires MLJ.jl version 0.19.3 or higher
```

We first define a logger, providing the API address of our running MLflow
instance. The experiment name and artifact location are optional.

```julia
logger = MLJFlow.Logger(
    "http://127.0.0.1:5000/api";
    experiment_name="MLJFlow test",
    artifact_location="./mlj-test"
)
```

Next, grab some synthetic data and choose an MLJ model:

```julia
X, y = make_moons(100) # a table and a vector with 100 rows
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model = DecisionTreeClassifier(max_depth=4)
```

Now we call `evaluate` as usual but provide the `logger` as a keyword argument:

```julia
evaluate(model, X, y, resampling=CV(nfolds=5), measures=[LogLoss(), Accuracy()], logger=logger)
```

Navigate to "http://127.0.0.1:5000" on your browser and select the "Experiment" matching
the name above ("MLJFlow test"). Select the single run displayed to see the logged results
of the performance evaluation.


### Saving and retrieving trained machines as MLflow artifacts

Let's train the model on all data and save the trained machine as an MLflow artifact:

```julia
mach = machine(model, X, y) |> fit!
run = MLJBase.save(logger, mach)
```

Notice that in this case `MLJBase.save` returns a run (and instance of `MLFlowRun` from
MLFlowClient.jl). 

To retrieve an artifact we need to use the MLFlowClient.jl API, and for that we need to
know the MLflow service that our `logger` wraps:

```julia
service = MLJFlow.service(logger)
```

And we reconstruct our trained machine thus:

```julia
using MLFlowClient
artifacts = MLFlowClient.listartifacts(service, run)
mach2 = machine(artifact[1].filepath)
```

We can predict using the deserialized machine:

```julia
predict(mach2, X)
```
