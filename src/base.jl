function log_evaluation(logger::MLFlowLogger, performance_evaluation)
    experiment = getorcreateexperiment(logger.client, logger.experiment_name;
        artifact_location=logger.artifact_location)
    model_name = name(performance_evaluation.model)
    run = createrun(logger.client, experiment;
        run_name="$(model_name) run")

    logmodelparams(logger.client, run, performance_evaluation.model)
    logmachinemeasures(logger.client, run, performance_evaluation.measure,
                        performance_evaluation.measurement)

    updaterun(logger.client, run, "FINISHED")
end

function save(logger::MLFlowLogger, mach::Machine)
    io = IOBuffer()
    save(io, mach)
    seekstart(io)

    model = mach.model
    model_name = name(model)

    experiment = getorcreateexperiment(logger.client, logger.experiment_name,
        artifact_location=logger.artifact_location)
    run = createrun(logger.client, experiment;
        run_name="$(model_name) run")

    logmodelparams(logger.client, run, model)
    fname = "$(model_name).jls"
    logartifact(logger.client, run, fname, io)
    updaterun(logger.client, run, "FINISHED")
end
