@testset "Solutions preserve SciML parameter access" begin
    opt = SnoptOptimizer()
    f = OptimizationFunction((x, p) -> sum(abs2, x), AutoForwardDiff())
    cache = init(OptimizationProblem(f, [1.0], [2.0]), opt)
    sol = SciMLBase.build_solution(cache, opt, [0.0], 0.0)
    @test sol.ps[1] == 2.0
end

@testset "Additional option names ignore case and repeated whitespace" begin
    @test_throws ArgumentError SnoptOptimizer(
        additional_options = Dict("major   Iterations LIMIT" => 2)
    )
    @test_throws ArgumentError SnoptOptimizer(
        additional_options = Dict(
            "Linesearch tolerance" => 0.9, "LINESEARCH TOLERANCE" => 0.8
        )
    )
    opt = SnoptOptimizer(additional_options = Dict("LINESEARCH   TOLERANCE" => 0.9))
    @test opt.additional_options == Dict("Linesearch tolerance" => 0.9)
end
