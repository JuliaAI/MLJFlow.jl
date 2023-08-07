function logmodelparams(client::MLFlow, run::MLFlowRun, model::Model)
    model_params = params(model) |> flat_params |> collect
    for (name, value) in model_params
        logparam(client, run, name, value)
    end
end

function logmachinemeasures(client::MLFlow, run::MLFlowRun, measures,
    measurements)
    measure_names = measures .|> info .|> x -> x.name
    for (name, value) in zip(measure_names, measurements)
        logmetric(client, run, name, value)
    end
end

"""
    runs(logger::MLFlowLogger)

Return a list of runs for the experiment specified by
`logger.experiment_name`. The list is returned as a
`Vector{MLFlowRun}`.
"""
runs(logger::MLFlowLogger) = searchruns(logger.client,
    getexperiment(logger.client, logger.experiment_name))
