@testset "good_name" begin
    @test MLJFlow.good_name(rms) == "RootMeanSquaredError"
    @test MLJFlow.good_name(macro_f1score) == "MulticlassFScore-beta_1.0"
    @test MLJFlow.good_name(log_score) == "LogScore-tol_2.22045e-16"
end

true
