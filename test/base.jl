@testset "logging functions" begin
    logger = MLFlowLogger("http://localhost:5000";
        experiment_name="MLJFlow tests",
        artifact_location="./mlj-test")

    X, y = make_moons(100)
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

    dtc = DecisionTreeClassifier()
    dtc_machine = machine(dtc, X, y)
    e1 = evaluate!(dtc_machine, resampling=CV(),
        measures=[LogLoss(), Accuracy()], verbosity=1, logger=logger)

    @testset "log_evaluation" begin
        runs = searchruns(logger.client,
            getexperiment(logger.client, logger.experiment_name))
        @test typeof(runs[1]) == MLFlowRun
    end

    @testset "save" begin
        run = MLJ.save(logger, dtc_machine)
        @test typeof(run) == MLFlowRun
        @test listartifacts(logger.client, run) |> length == 1
    end

    experiment = getorcreateexperiment(logger.client, logger.experiment_name)
    deleteexperiment(logger.client, experiment)
end
