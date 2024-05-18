"""
    Logger(apiroot; experiment_name="MLJ experiment",
        artifact_location=nothing)

A wrapper around a MLFlow service, with an experiment name and an artifact
location. This is the type passed to the `logger` keyword argument of
multiple methods in MLJBase.

To use this logger, you need to have a MLFlow server running. For more
information, see [MLFlow documentation](https://www.mlflow.org/docs/latest/quickstart.html).

Depending on the MLFlow server configuration, the `apiroot` can be a local
or a remote server API. The `experiment_name` is used to identify the
experiment in the MLFlow server. If the experiment does not exist, it will be
created with the default name "MLJ experiment". The `artifact_location` is
used to store the artifacts of the experiment. If not provided, a default
artifact location will be defined by MLFlow. For more information, see
[MLFlow documentation](https://www.mlflow.org/docs/latest/tracking.html).

This constructor returns a `Logger` object, containing the experiment
name and the artifact location specified previously. Also it contains a
`MLFlow` service, which is used to communicate with the MLFlow server. For
more information, see [MLFlowClient.jl](https://juliaai.github.io/MLFlowClient.jl/dev/reference/#MLFlowClient.MLFlow).

"""
struct Logger
    service::MLFlow
    verbosity::Int
    experiment_name::String
    artifact_location::Union{String,Nothing}
end
function Logger(apiroot; experiment_name="MLJ experiment",
    artifact_location=nothing, verbosity=1)
    service = MLFlow(apiroot)
    @async process_queue()

    Logger(service, verbosity, experiment_name, artifact_location)
end
function show(io::IO, logger::MLJFlow.Logger)
    print(io,
        "MLFLowLogger(\"$(logger.service.apiroot)\",\n" *
        "   experiment_name=\"$(logger.experiment_name)\",\n" *
        "   artifact_location=\"$(logger.artifact_location)\",\n" *
        ") using MLFlow API version $(logger.service.apiversion)"
    )
end
