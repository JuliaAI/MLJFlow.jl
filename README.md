# MLJFlow.jl

| Branch | Build | Coverage |
| :---: | :---: | :---: |
| dev | [![Continuous Integration (CPU)][ci-dev-img]][ci-dev] | [![Code Coverage][codecov-dev-img]][codecov-dev] |

[ci-dev]: https://github.com/pebeto/MLJFlow.jl/actions/workflows/CI.yml
[ci-dev-img]: https://github.com/pebeto/MLJFlow.jl/actions/workflows/CI.yml/badge.svg?branch=dev "Continuous Integration (CPU)"
[codecov-dev]: https://codecov.io/github/JuliaAI/MLJFlow.jl
[codecov-dev-img]: https://codecov.io/github/JuliaAI/MLJFlow.jl/graph/badge.svg?token=TBCMJOK1WR "Code Coverage"

[MLJ](https://github.com/alan-turing-institute/MLJ.jl) is a Julia framework for
combining and tuning machine learning models. MLJFlow is a package that extends
the MLJ capabilities to use [mlflow](https://mlflow.org/) as a backend for
model tracking and experiment management.

## Background

This project was launched as part of the GSoC 2023 program. The proposal description can be
found [here](https://summerofcode.withgoogle.com/programs/2023/projects/iRxuzeGJ).
The entire workload is divided into three different repositories:
[MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl),
[MLFlowClient.jl](https://github.com/JuliaAI/MLFlowClient.jl) and this one.

## Features

- [x] [mlflow](https://mlflow.org) cycle automation (create experiment, create run, log metrics, log parameters,
      log artifacts, etc.)

- [x] Provides a wrapper `Logger` for MLFlowClient.jl clients and associated metadata;
      instances of this type are valid "loggers", which can be passed to MLJ functions
      supporting the `logger` keyword argument.

- [x] Provides [mlflow](https://mlflow.org) integration with MLJ's `evaluate!`/`evaluate`
      method (model **performance evaluation**)

- [x] Extends MLJ's `MLJ.save` method, to save trained machines as retrievable
      [mlflow](https://mlflow.org) client artifacts

- [x] Provides [mlflow](https://mlflow.org) integration with MLJ's `TunedModel` wrapper
      (to log **hyper-parameter tuning** workflows)

- [ ] Provides [mlflow](https://mlflow.org) integration with MLJ's `IteratedModel` wrapper
      (to log **controlled iteration** of tree gradient boosters, neural networks, and
      other iterative models)

- [x] Plays well with **composite models** (pipelines, stacks, etc.)


## Examples

### Logging a model performance evaluation

The example below assumes the user is familiar with basic [mlflow](https://mlflow.org)
concepts. We suppose an [mlflow](https://mlflow.org) API service is running on a local
server, with address "http://127.0.0.1:5000"; your actual URI may differ.  In a
shell/console, you run `mlflow server` to launch an [mlflow](https://mlflow.org) service
on a local server.

**Security issues**. Recent versions of [mlflow](https://mlflow.org) require you to
configure security to allow you to navigate to the service in your browser. Otherwise you
will encounter a 403 Error.  Alternatively, you can, at your own risk, use the unsafe
nuclear option `mlflow server --disable-security-middleware`.

Refer to the [mlflow](https://mlflow.org) documentation for necessary background.

**Important.** For the examples that follow, we assume `MLJ`, `MLJDecisionTreeClassifier`
and `MLFlowClient` are in the user's active Julia package environment. Previous versions
of MLJ included MLJFlow as a dependency; since MLJ 0.23, you must explicitly import
MLJFlow:

```julia
    using MLJ # Requires MLJ.jl version 0.23 or higher
	using MLJFlow
```

We begin by defining a logger, providing the API address of our running [mlflow](https://mlflow.org) instance
(possibly different from the one below). The experiment name and artifact
location are optional.

```julia
logger = MLJFlow.Logger(
    "http://127.0.0.1:5000";
    experiment_name="MLJFlow test",
    artifact_location="./mlj-test"
)
```

Next, we grab some synthetic data and choose an MLJ model:

```julia
X, y = make_moons(100) # a table and a vector with 100 rows
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model = DecisionTreeClassifier(max_depth=4)
```

Now we call `evaluate` as usual but provide the `logger` as a keyword argument:

```julia
evaluate(
    model,
    X,
    y,
    resampling=CV(nfolds=5),
    measures=[LogLoss(), Accuracy()],
    logger=logger,
)
```

Navigate to "http://127.0.0.1:5000" on your browser and select the "Experiment Name"
matching the name above ("MLJFlow test"). See "Security issues" above if you get a 403
error. We are interested in "Evaluation runs", which we can navigate to in mljflow v3.11
by selecting "Evaluation runs" in the left sidebar. Select the single run displayed to see
the logged results of the performance evaluation.


### Logging outcomes of model tuning

Continuing with the previous example:

```julia
r = range(model, :max_depth, lower=1, upper=5)
tmodel = TunedModel(
    model,
    tuning=Grid(),
    range = r;
    resampling=CV(nfolds=9),
    measures=[LogLoss(), Accuracy()],
    logger=logger,
)

mach = machine(tmodel, X, y) |> fit!
```

Return to the browser page (refreshing if necessary) and you will find five more
performance evaluations logged, one for each value of `max_depth` evaluated in tuning.


### Saving and retrieving trained machines as [mlflow](https://mlflow.org) artifacts

Let's train the model on all data and save the trained machine as an [mlflow](https://mlflow.org) artifact:

```julia
mach = machine(model, X, y) |> fit!
run = MLJ.save(logger, mach)
```

Notice that in this case `MLJBase.save` returns a run (an instance of `Run` from
MLFlowClient.jl).

To retrieve the saved machine:

```julia
mach2 = MLJFlow.load(logger, run)
```

We can predict using the deserialized machine:

```julia
predict(mach2, X)
```

### Setting a global logger

Set `logger` as the global logging target by running `default_logger(logger)`. Then,
unless explicitly overridden, all loggable workflows will log to `logger`. In particular,
to *suppress* logging, you will need to specify `logger=nothing` in your calls.

So, for example, if we run the following setup

```julia
using MLJ

# using a new experiment name here:
logger = MLJFlow.Logger(
    "http://127.0.0.1:5000/api";
    experiment_name="test global logging",
    artifact_location="./mlj-test"
)

default_logger(logger)

X, y = make_moons(100) # a table and a vector with 100 rows
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model = DecisionTreeClassifier()
```

Then the following is automatically logged

```julia
evaluate(model, X, y)
```

But the following is *not* logged:


```julia
evaluate(model, X, y; logger=nothing)
```

To save a machine when a default logger is set, one can use the following syntax:

```julia
mach = machine(model, X, y) |> fit!
MLJ.save(mach)
```

Retrieve the saved machine as described earlier.
