@testset "logging tuned models" begin
    model = DecisionTreeClassifier()

    service = MLJFlow.service(LOGGER)

    r = range(model, :max_depth, lower=1, upper=5,
        scale=:linear);
    self_tuning_tree = TunedModel(model=model, tuning=Grid(), range=r,
        measure=LogLoss(), logger=LOGGER)

    tuning_mach = machine(self_tuning_tree, X, y)
    e1 = evaluate!(tuning_mach, resampling=Holdout(), verbosity=1)

    runs = searchruns(service, getexperiment(service,
        LOGGER.experiment_name))

    max_depth_values = runs .|> (x -> x.data.params["max_depth"].value)
    @test issubset(r.lower:r.upper |> collect, parse.(Int, max_depth_values))

    experiment = getorcreateexperiment(service, LOGGER.experiment_name)
    deleteexperiment(service, experiment)
end
