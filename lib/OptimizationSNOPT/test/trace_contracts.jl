using OptimizationSNOPT
using Test

@testset "Trace frequencies reject Boolean and nonpositive values" begin
    for constructor in (SnoptTraceMinimal, SnoptTraceAll)
        @test_throws ArgumentError constructor(print_frequency = true)
        @test_throws ArgumentError constructor(store_frequency = false)
        @test_throws ArgumentError constructor(true)
    end
    @test_throws ArgumentError SnoptTraceMinimal(0)
end

@testset "Trace frequencies accept positive Integer values" begin
    for constructor in (SnoptTraceMinimal, SnoptTraceAll)
        level = constructor(print_frequency = Int32(2), store_frequency = Int8(3))
        @test level.print_frequency == 2
        @test level.store_frequency == 3
    end
end
