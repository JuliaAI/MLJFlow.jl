function logmodelparams(client::MLFlow, run::MLFlowRun, model::Model)
    model_params = deep_params(model) |> flat_params |> collect
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
