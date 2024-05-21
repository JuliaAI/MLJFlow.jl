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
    _LOGGING_CHANNEL::Channel{Tuple}
end

function Logger(apiroot; experiment_name="MLJ experiment",
    artifact_location=nothing, verbosity=1)
    service = MLFlow(apiroot)
    LOGGING_CHANNEL = open_logging_channel()

    Logger(service, verbosity, experiment_name, artifact_location, LOGGING_CHANNEL)
end

function show(io::IO, logger::MLJFlow.Logger)
    print(io,
        "MLFLowLogger(\"$(logger.service.apiroot)\",\n" *
        "   experiment_name=\"$(logger.experiment_name)\",\n" *
        "   artifact_location=\"$(logger.artifact_location)\",\n" *
        ") using MLFlow API version $(logger.service.apiversion)"
    )
end

"""
    close(logger::Logger)

Each logger instance has a background loop that allows to execute the logging
operations from the `_LOGGING_TASKS_CHANNEL`. This function closes the channel
to stop the background loop.
"""
function close(logger::Logger)
    close(logger._LOGGING_CHANNEL)
end

"""
    open_logging_channel(logger::Logger)

To allow safe concurrent logging operations, this function opens the
`_LOGGING_TASKS_CHANNEL` of the logger and starts a background worker.
"""
function open_logging_channel()
    LOGGING_CHANNEL = Channel{Tuple}()

    # NOTE: This background loop allows to execute the logging operations from
    # the LOGGING_TASKS_CHANNEL. The execution result is sent back to the
    # requesting thread through the result_channel.
    # Until May 2024, mlflow does not support concurrent experiment creation,
    # which does not allow to run the logging operations in multi-threading and
    # multi-processing.
    #
    # Its usage can be seen in the `log_evaluation` function in `base.jl`.
    Threads.@spawn for (logging_function, logger, performance_evaluation, result_channel) in LOGGING_CHANNEL
        result = logging_function(logger, performance_evaluation)
        put!(result_channel, result)
    end

    return LOGGING_CHANNEL
end
