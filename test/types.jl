@testset "logger type" begin
    logger = MLFlowLogger("http://localhost:5000")

    @test typeof(logger) == MLFlowLogger
    @test typeof(logger.client) == MLFlow
end
