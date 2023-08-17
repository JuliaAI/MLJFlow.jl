function log_evaluation(logger::MLFlowLogger, performance_evaluation)
    experiment = getorcreateexperiment(logger.service, logger.experiment_name;
        artifact_location=logger.artifact_location)
    model_name = name(performance_evaluation.model)
    run = createrun(logger.service, experiment;
        run_name="$(model_name) run")

    logmodelparams(logger.service, run, performance_evaluation.model)
    logmachinemeasures(logger.service, run, performance_evaluation.measure,
                        performance_evaluation.measurement)

    updaterun(logger.service, run, "FINISHED")
end

function save(logger::MLFlowLogger, mach::Machine)
    io = IOBuffer()
    save(io, mach)
    seekstart(io)

    model = mach.model
    model_name = name(model)

    experiment = getorcreateexperiment(logger.service, logger.experiment_name,
        artifact_location=logger.artifact_location)
    run = createrun(logger.service, experiment;
        run_name="$(model_name) run")

    logmodelparams(logger.service, run, model)
    fname = "$(model_name).jls"
    logartifact(logger.service, run, fname, io)
    updaterun(logger.service, run, "FINISHED")
end

"""
    service(logger::MLFlowLogger)

Returns the MLFlow service of a logger.
"""
service(logger::MLFlowLogger) = logger.service
