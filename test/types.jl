@testset "types" begin
    logger = MLJFlow.Logger(ENV["MLFLOW_URI"])

    @test typeof(logger) == MLJFlow.Logger
    @test typeof(logger.service) == MLFlow

    io = IOBuffer()
    show(io, logger)
    test_string = "MLFLowLogger(\"$(logger.service.baseuri)\",\n" *
        "   experiment_name=\"$(logger.experiment_name)\",\n" *
        "   artifact_location=\"$(logger.artifact_location)\",\n" *
        ") using MLFlow API version $(logger.service.apiversion)"
    @test String(take!(io)) == test_string
end
