"""
    logmodelparams(service::MLFlow, run::MLFlowRun, model::Model)

Extracts the parameters of a model and logs them to the MLFlow server.
The information coming from `flat_params` is in the form of a NamedTuple, with
each key as a summary of the parameters from parents to children separated by
underscores.

# Arguments
- `service::MLFlow`: An MLFlow service. See [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlow)
- `run::MLFlowRun`: An MLFlow run. See [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlowRun)
- `model::Model`: A MLJ model.
"""
function logmodelparams(service::MLFlow, run::MLFlowRun, model::Model)
    model_params = flat_params(model)
    for key in keys(model_params)
        logparam(service, run, key, getproperty(model_params, key))
    end
end

const MLFLOW_CHAR_SET =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-. /"

"""
    good_name(measure)

**Private method.**

Returns a string representation of `measure` that can be used as a valid name in
MLflow. Includes the value of the first hyperparameter, if there is one.

```julia
julia> good_name(macro_f1score)
"MulticlassFScore-beta_1.0"

"""
function good_name(measure)
    name = string(measure)
    name = replace(name, ", …" => "")
    name = replace(name, " = " => "_")
    name = replace(name, "()" => "")
    name = replace(name, ")" => "")
    map(collect(name)) do char
        char in ['(', ','] && return '-'
        char == '=' && return '_'
        char in MLFLOW_CHAR_SET && return char
        " "
    end |> join
end

"""
    logmachinemeasures(service::MLFlow, run::MLFlowRun, model::Model)

Extracts the parameters of a model and logs them to the MLFlow server.

# Arguments

- `service::MLFlow`: An MLFlow service. See
  [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlow)

- `run::MLFlowRun`: An MLFlow run. See
  [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlowRun)

- `measures`: A vector of measures.

- `measurements`: A vector of measurements.

"""
function logmachinemeasures(service::MLFlow, run::MLFlowRun, measures,
    measurements)
    measure_names = measures .|> good_name
    for (name, value) in zip(measure_names, measurements)
        logmetric(service, run, name, value)
    end
end

"""
    service(logger::MLFlowLogger)

Returns the MLFlow service of a logger.
"""
service(logger::MLFlowLogger) = logger.service
