function log_evaluation_task(logger::Logger, performance_evaluation)
    experiment = getorcreateexperiment(logger.service, logger.experiment_name;
        artifact_location=logger.artifact_location)
    run = createrun(logger.service, string(experiment.experiment_id);
        tags=[
            Tag("resampling", string(performance_evaluation.resampling)),
            Tag("repeats", string(performance_evaluation.repeats)),
            Tag("model type", name(performance_evaluation.model)),
        ]
    )

    logmodelparams(logger.service, run, performance_evaluation.model)
    logmachinemeasures(logger.service, run, performance_evaluation.measure,
                        performance_evaluation.measurement)

    updaterun(logger.service, run; status=RunStatus.FINISHED, run_name=missing)
    return run
end

function log_evaluation(logger::Logger, performance_evaluation)
    result_channel = Channel{Any}(1)

    put!(logger.logging_channel, (log_evaluation_task, logger,
        performance_evaluation, result_channel))

    result = take!(result_channel)
    result isa CapturedException && throw(result)
    return result
end

function save(logger::Logger, machine:: Machine)
    io = IOBuffer()
    save(io, machine)
    seekstart(io)

    model = machine.model

    experiment = getorcreateexperiment(logger.service, logger.experiment_name;
        artifact_location=logger.artifact_location)
    run = createrun(logger.service, string(experiment.experiment_id))

    logmodelparams(logger.service, run, model)

    # Upload artifact via the mlflow-artifacts proxy.
    # Strip the scheme prefix from artifact_uri to get the proxy-relative path.
    artifact_path = replace(run.info.artifact_uri, r"^[a-z\-]+:/*" => "")
    uploadartifact(logger.service, "$artifact_path/machine.jls", take!(io))

    updaterun(logger.service, run; status=RunStatus.FINISHED, run_name=missing)
    return run
end
