@testset "logging functions" begin
    logger = MLFlowLogger("http://localhost:5000";
        experiment_name="MLJFlow tests",
        artifact_location="./mlj-test")

    X, y = make_moons(100)
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

    pipe = Standardizer() |> DecisionTreeClassifier()
    mach = machine(pipe, X, y)
    e1 = evaluate!(mach, resampling=CV(),
        measures=[LogLoss(), Accuracy()], verbosity=1, logger=logger)

    @testset "log_evaluation" begin
        runs = searchruns(logger.client,
            getexperiment(logger.client, logger.experiment_name))
        @test typeof(runs[1]) == MLFlowRun
    end

    @testset "ensuring logging" begin
        runs = searchruns(logger.client,
            getexperiment(logger.client, logger.experiment_name))
        @test issetequal(keys(runs[1].data.params),
            String.([keys(MLJModelInterface.flat_params(pipe))...]))
    end

    @testset "save" begin
        run = MLJBase.save(logger, mach)
        @test typeof(run) == MLFlowRun

        artifacts = listartifacts(logger.client, run)
        @test artifacts |> length == 1

        loaded_mach = machine(artifacts[1].filepath)
        @test loaded_mach.model isa ProbabilisticPipeline
    end

    experiment = getorcreateexperiment(logger.client, logger.experiment_name)
    deleteexperiment(logger.client, experiment)
end
