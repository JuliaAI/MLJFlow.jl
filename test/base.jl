@testset verbose = true "base" begin
    logger = MLJFlow.Logger(ENV["MLFLOW_TRACKING_URI"];
        experiment_name="MLJFlow tests",
        artifact_location="/tmp/mlj-test")

    X, y = make_moons(100)
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

    pipe = Standardizer() |> DecisionTreeClassifier()
    mach = machine(pipe, X, y)
    e1 = evaluate!(mach, resampling=CV(nfolds=3),
        measures=[LogLoss(), Accuracy()], verbosity=1, logger=logger)

    @testset "log_evaluation" begin
        experiment = getexperimentbyname(logger.service, logger.experiment_name)
        runs, _ = searchruns(logger.service;
            experiment_ids=[string(experiment.experiment_id)])
        @test typeof(runs[1]) == Run
    end

    @testset "ensuring logging" begin
        experiment = getexperimentbyname(logger.service, logger.experiment_name)
        runs, _ = searchruns(logger.service;
            experiment_ids=[string(experiment.experiment_id)])
        @test issetequal([p.key for p in runs[1].data.params],
            String.([keys(MLJModelInterface.flat_params(pipe))...]))
    end

    @testset "save" begin
        run = MLJBase.save(logger, mach)
        @test typeof(run) == Run

        loaded_mach = MLJFlow.load(logger, run)
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
        experiment = getexperimentbyname(logger.service, logger.experiment_name)
        runs, _ = searchruns(logger.service;
            experiment_ids=[string(experiment.experiment_id)])
        @test any(r -> isempty(r.data.params), runs)
    end

    experiment = MLJFlow.getorcreateexperiment(logger.service, logger.experiment_name)
    deleteexperiment(logger.service, string(experiment.experiment_id))
end
