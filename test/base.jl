@testset verbose = true "base" begin
    logger = MLJFlow.Logger(ENV["MLFLOW_URI"];
        experiment_name="MLJFlow tests",
        artifact_location="/tmp/mlj-test")

    X, y = make_moons(100)
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

    pipe = Standardizer() |> DecisionTreeClassifier()
    mach = machine(pipe, X, y)
    e1 = evaluate!(mach, resampling=CV(),
        measures=[LogLoss(), Accuracy()], verbosity=1, logger=logger)

    @testset "log_evaluation" begin
        runs = searchruns(logger.service,
            getexperiment(logger.service, logger.experiment_name))
        @test typeof(runs[1]) == MLFlowRun
    end

    @testset "ensuring logging" begin
        runs = searchruns(logger.service,
            getexperiment(logger.service, logger.experiment_name))
        @test issetequal(keys(runs[1].data.params),
            String.([keys(MLJModelInterface.flat_params(pipe))...]))
    end

    @testset "save" begin
        run = MLJBase.save(logger, mach)
        @test typeof(run) == MLFlowRun

        artifacts = listartifacts(logger.service, run)
        @test artifacts |> length == 1

        loaded_mach = machine(artifacts[1].filepath)
        @test loaded_mach.model isa ProbabilisticPipeline

        test_x, test_y = make_moons(1)
        pred = predict(mach, test_x)[1]
        loaded_mach_pred = predict(loaded_mach, test_x)[1]
        @test pdf(pred, 0) == pdf(loaded_mach_pred, 0)
        @test pdf(pred, 1) == pdf(loaded_mach_pred, 1)
    end

    @testset "accesor methods" begin
        @test MLJFlow.service(logger) isa MLFlow
    end

    @testset "log_evaluation_with_zero_param_model" begin
        zeroparams_machine = machine(ConstantClassifier(), X, y)

        e1 = evaluate!(zeroparams_machine, resampling=CV(),
            measures=[LogLoss(), Accuracy()], verbosity=1, logger=logger)
        runs = searchruns(logger.service,
            getexperiment(logger.service, logger.experiment_name))
        @test isempty(runs[3].data.params)
    end

    experiment = getorcreateexperiment(logger.service, logger.experiment_name)
    deleteexperiment(logger.service, experiment)
end
