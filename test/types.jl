@testset "types" begin
    logger = MLJFlow.Logger(ENV["MLFLOW_URI"])

    @test typeof(logger) == MLJFlow.Logger
    @test typeof(logger.service) == MLFlow
end
