@testset verbose = true "base" begin
    model = DecisionTreeClassifier()
    pipe = Standardizer() |> model
    mach = machine(pipe, X, y)
    e1 = evaluate!(mach, resampling=CV(),
        measures=[LogLoss(), Accuracy()], verbosity=1, logger=LOGGER)

    service = MLJFlow.service(LOGGER)

    @testset "accesor methods" begin
        @test service isa MLFlow
    end

    @testset "log_evaluation" begin
        runs = searchruns(service, getexperiment(service,
            LOGGER.experiment_name))
        @test typeof(runs[1]) == MLFlowRun
    end

    @testset "ensuring logging" begin
        runs = searchruns(service, getexperiment(service,
            LOGGER.experiment_name))
        @test issetequal(keys(runs[1].data.params),
            String.([keys(MLJModelInterface.flat_params(pipe))...]))
    end

    @testset "save" begin
        run = MLJBase.save(LOGGER, mach)
        @test typeof(run) == MLFlowRun

        artifacts = listartifacts(service, run)
        @test artifacts |> length == 1

        loaded_mach = machine(artifacts[1].filepath)
        @test loaded_mach.model isa ProbabilisticPipeline

        test_x, test_y = make_moons(1)
        pred = predict(mach, test_x)[1]
        loaded_mach_pred = predict(loaded_mach, test_x)[1]
        @test pdf(pred, 0) == pdf(loaded_mach_pred, 0)
        @test pdf(pred, 1) == pdf(loaded_mach_pred, 1)
    end

    experiment = getorcreateexperiment(service, LOGGER.experiment_name)
    deleteexperiment(service, experiment)
end
