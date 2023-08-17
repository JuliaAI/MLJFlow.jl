@testset "types" begin
    logger = MLFlowLogger("http://localhost:5000")

    @test typeof(logger) == MLFlowLogger
    @test typeof(logger.service) == MLFlow
end
