"""
    MLFlowLogger(baseuri; experiment_name="MLJ experiment",
        artifact_location=nothing)

A wrapper around a MLFlow client, with an experiment name and an artifact
location. This is the type passed to the `logger` keyword argument of
multiple methods in MLJBase.

# Fields
- `client`: an MLFlow client(see [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlow))
- `experiment_name`: the name of the experiment. If not provided, a default
experiment with the name "MLJ experiment" will be created.
- `artifact_location`: the location of the artifact store.  If not provided,
a default artifact location will be defined by MLFlow. For more information,
see [MLFlow documentation](https://www.mlflow.org/docs/latest/tracking.html).

# Return value
A `MLFlowLogger` object, containing the client, the experiment name and the
artifact location.

"""
struct MLFlowLogger
    client::MLFlow
    experiment_name::String
    artifact_location::Union{String,Nothing}
end
function MLFlowLogger(baseuri; experiment_name="MLJ experiment",
    artifact_location=nothing)
    client = MLFlow(baseuri)
    MLFlowLogger(client, experiment_name, artifact_location)
end
