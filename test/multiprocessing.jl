@testset verbose = true "multiprocessing" begin
    logger = MLJFlow.Logger(ENV["MLFLOW_TRACKING_URI"];
        experiment_name="MLJFlow multiprocessing tests",
        artifact_location="/tmp/mlj-test")

    X, y = make_moons(100)
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

    model = DecisionTreeClassifier()
    r = range(model, :max_depth, lower=1, upper=6)

    function test_tuned_model(acceleration_method)
        experiment = MLJFlow.getorcreateexperiment(logger.service, logger.experiment_name)
        existing_runs, _ = searchruns(logger.service;
            experiment_ids=[string(experiment.experiment_id)])
        n_before = length(existing_runs)

        tuned_model = TunedModel(
            model=model,
            range=r,
            logger=logger,
            acceleration=acceleration_method,
            n=10,
        )
        tuned_model_mach = machine(tuned_model, X, y)
        fit!(tuned_model_mach)

        runs, _ = searchruns(logger.service;
            experiment_ids=[string(experiment.experiment_id)])

        @assert length(runs) - n_before == 10

        deleteexperiment(logger.service, string(experiment.experiment_id))
    end

    @testset "log_evaluation_with_cpu_threads" begin
        test_tuned_model(CPUThreads())
    end

    @testset "log_evaluation_with_cpu_processes" begin
        test_tuned_model(CPUProcesses())
    end
end
