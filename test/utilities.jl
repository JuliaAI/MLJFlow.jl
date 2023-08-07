@testset "flattening parameters" begin
    t = (a = (ax = (ax1 = 1, ax2 = 2), ay = 3), b = 4)
    dict_t = Dict(
        "a__ax__ax1" => 1,
        "a__ax__ax2" => 2,
        "a__ay" => 3,
        "b" => 4,
    )
    @test flat_params(t) == dict_t
end
