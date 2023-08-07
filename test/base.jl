@testset "logging functions" begin
    logger = MLFlowLogger("http://localhost:5000";
        experiment_name="MLJFlow tests",
        artifact_location="./mlj-test")

    X, y = make_moons(100)
    clf = ConstantClassifier()
    clf_machine = machine(clf, X, y)
    e1 = evaluate!(clf_machine, resampling=CV(),
        measures=[LogLoss(), Accuracy()], verbosity=1)

    @testset "log_evaluation" begin
        run = log_evaluation(logger, e1)
        @test typeof(run) == MLFlowRun
    end

    @testset "save" begin
        run = save(logger, clf_machine)
        @test typeof(run) == MLFlowRun
        @test listartifacts(logger.client, run) |> length == 1
    end

    experiment = getorcreateexperiment(logger.client, logger.experiment_name)
    deleteexperiment(logger.client, experiment)
end
