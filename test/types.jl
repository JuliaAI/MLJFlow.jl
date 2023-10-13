@testset "types" begin
    logger = MLJFlow.Logger("http://localhost:5000")

    @test typeof(logger) == MLJFlow.Logger
    @test typeof(logger.service) == MLFlow
end
