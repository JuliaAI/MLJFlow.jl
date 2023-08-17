"""
    MLFlowLogger(baseuri; experiment_name="MLJ experiment",
        artifact_location=nothing)

A wrapper around a MLFlow service, with an experiment name and an artifact
location. This is the type passed to the `logger` keyword argument of
multiple methods in MLJBase.

To use this logger, you need to have a MLFlow server running. For more
information, see [MLFlow documentation](https://www.mlflow.org/docs/latest/quickstart.html).

Depending on the MLFlow server configuration, the `baseuri` can be a local
server or a remote server. The `experiment_name` is used to identify the
experiment in the MLFlow server. If the experiment does not exist, it will be
created with the default name "MLJ experiment". The `artifact_location` is
used to store the artifacts of the experiment. If not provided, a default
artifact location will be defined by MLFlow. For more information, see
[MLFlow documentation](https://www.mlflow.org/docs/latest/tracking.html).

This constructor returns a `MLFlowLogger` object, containing the experiment
name and the artifact location specified previously. Also it contains a
`MLFlow` service, which is used to communicate with the MLFlow server. For
more information, see [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlow).

"""
struct MLFlowLogger
    service::MLFlow
    verbosity::Integer
    experiment_name::String
    artifact_location::Union{String,Nothing}
end
function MLFlowLogger(baseuri; experiment_name="MLJ experiment",
    artifact_location=nothing, verbosity=1)
    service = MLFlow(baseuri)

    if ~healthcheck(service)
        error("It seems that the MLFlow server is not running. For more information, see https://mlflow.org/docs/latest/quickstart.html")
    end
    MLFlowLogger(service, 1, experiment_name, artifact_location)
end
