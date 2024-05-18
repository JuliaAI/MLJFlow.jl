@testset verbose = true "multiprocessing" begin
    logger = MLJFlow.Logger(ENV["MLFLOW_TRACKING_URI"];
        experiment_name="MLJFlow multiprocessing tests",
        artifact_location="/tmp/mlj-test")

    X, y = make_moons(100)
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

    model = DecisionTreeClassifier()
    r = range(model, :max_depth, lower=1, upper=6)

    function test_tuned_model(acceleration_method)
        tuned_model = TunedModel(
            model=model,
            range=r,
            logger=logger,
            acceleration=acceleration_method,
            n=100,
        )
        tuned_model_mach = machine(tuned_model, X, y)
        fit!(tuned_model_mach)

        experiment = getorcreateexperiment(logger.service, logger.experiment_name)
        runs = searchruns(logger.service, experiment)

        @assert length(runs) == 100

        deleteexperiment(logger.service, experiment)
    end

    @testset "log_evaluation_with_cpu_threads" begin
        test_tuned_model(CPUThreads())
    end

    @testset "log_evaluation_with_cpu_processes" begin
        test_tuned_model(CPUProcesses())
    end
end
