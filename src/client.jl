"""
    logmodelparams(client::MLFlow, run::MLFlowRun, model::Model)

Extracts the parameters of a model and logs them to the MLFlow server.
The information coming from `flat_params` is in the form of a NamedTuple, with
each key as a summary of the parameters from parents to children separated by
underscores.

# Arguments
- `client::MLFlow`: An MLFlow client. See [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlow)
- `run::MLFlowRun`: An MLFlow run. See [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlowRun)
- `model::Model`: A MLJ model.
"""
function logmodelparams(client::MLFlow, run::MLFlowRun, model::Model)
    model_params = flat_params(model)
    for key in keys(model_params)
        logparam(client, run, key, getproperty(model_params, key))
    end
end

"""
    logmachinemeasures(client::MLFlow, run::MLFlowRun, model::Model)

Extracts the parameters of a model and logs them to the MLFlow server.

# Arguments
- `client::MLFlow`: An MLFlow client. See [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlow)
- `run::MLFlowRun`: An MLFlow run. See [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlowRun)
- `measures`: A vector of measures.
- `measurements`: A vector of measurements.
"""
function logmachinemeasures(client::MLFlow, run::MLFlowRun, measures,
    measurements)
    measure_names = measures .|> info .|> x -> x.name
    for (name, value) in zip(measure_names, measurements)
        logmetric(client, run, name, value)
    end
end
