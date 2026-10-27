using OptimizationBase, OptimizationSNOPT
using SciMLBase
using SNOPT
using Test
using ForwardDiff

@testset "SnoptOptimizer validation" begin
    @test_throws ArgumentError SnoptOptimizer(major_print_level = -1)
    @test_throws ArgumentError SnoptOptimizer(minor_print_level = -1)
    @test_throws ArgumentError SnoptOptimizer(major_iterations_limit = 0)
    @test_throws ArgumentError SnoptOptimizer(minor_iterations_limit = 0)
    @test_throws ArgumentError SnoptOptimizer(major_optimality_tolerance = 0.0)
    @test_throws ArgumentError SnoptOptimizer(major_feasibility_tolerance = 0.0)
    @test_throws ArgumentError SnoptOptimizer(minor_feasibility_tolerance = -1.0)
    @test_throws ArgumentError SnoptOptimizer(hessian = "full memory")
    @test_throws ArgumentError SnoptOptimizer(additional_options = ["Major step limit" => 2.0])
    @test_throws ArgumentError SnoptOptimizer(additional_options = Dict("" => 2.0))
    @test_throws ArgumentError SnoptOptimizer(additional_options = Dict("Major step limit" => true))
    @test_throws ArgumentError SnoptOptimizer(additional_options = Dict("Major step limit" => [2.0]))

    if SNOPT.has_snopt()
        @test_throws ArgumentError SnoptOptimizer(
            additional_options = Dict("Definitely unknown option" => 1)
        )
        @test_throws ArgumentError SnoptOptimizer(
            additional_options = Dict("Hessian" => "definitely invalid")
        )
    end

    # Valid combinations should construct without error.
    @test SnoptOptimizer(hessian = "limited_memory") isa SnoptOptimizer
    @test SnoptOptimizer(
        additional_options = Dict(
            "Major step limit" => 2.0,
            :linesearch_tolerance => 0.9,
            "Hessian" => "full memory",
        )
    ) isa SnoptOptimizer

    @testset "hessian = $(repr(hessian)) passes $(repr(expected))" for (hessian, expected) in (
            (nothing, String[]),
            ("full_memory", ["Hessian full memory"]),
            ("limited_memory", ["Hessian limited memory"]),
        )
        opt = SnoptOptimizer(; hessian)
        options = OptimizationSNOPT.snopt_optimizer_option_pairs(opt)
        @test [key for (key, _) in options if startswith(key, "Hessian")] == expected
    end
    @test SnoptOptimizer().hessian === nothing

    opt = SnoptOptimizer()
    @test !SciMLBase.requireshessian(opt)
    @test !SciMLBase.requiresconshess(opt)
    @testset "SNOPT status code $code maps to $retcode" for (code, retcode) in (
            (1, SciMLBase.ReturnCode.Success),
            (13, SciMLBase.ReturnCode.Infeasible),
            # 5 and 6 are elastic-mode results, in which the constraints may be violated.
            (5, SciMLBase.ReturnCode.Infeasible),
            (6, SciMLBase.ReturnCode.Infeasible),
            # Unbounded (SNOPT inform 21/22) must map to an existing ReturnCode.
            (21, SciMLBase.ReturnCode.Failure),
            (22, SciMLBase.ReturnCode.Failure),
            (31, SciMLBase.ReturnCode.MaxIters),
            (32, SciMLBase.ReturnCode.MaxIters),
            # 33 is SNOPT's "superbasics limit is too small": a sizing failure, not an
            # iteration limit.
            (33, SciMLBase.ReturnCode.Failure),
            (34, SciMLBase.ReturnCode.MaxTime),
            (73, SciMLBase.ReturnCode.Terminated),
        )
        @test OptimizationSNOPT.map_retcode(code) == retcode
    end
    @test OptimizationSNOPT.snopt_bound_value(Inf) == OptimizationSNOPT.SNOPT_BOUND_INF
    @test OptimizationSNOPT.snopt_bound_value(-Inf) == -OptimizationSNOPT.SNOPT_BOUND_INF
    @test OptimizationSNOPT.snopt_returned_without_evaluating(
        (status = 0, ws = (major_itns = 0, iterations = 0))
    )
    @test !OptimizationSNOPT.snopt_returned_without_evaluating(
        (status = 1, ws = (major_itns = 0, iterations = 0))
    )
    @test !OptimizationSNOPT.snopt_returned_without_evaluating(
        (status = 0, ws = (major_itns = 1, iterations = 2))
    )
end

# These checks run before the first call into the SNOPT library, so they hold
# without the library.
@testset "solve rejects invalid arguments with an ArgumentError" begin
    quad(x, p) = (x[1] - 1)^2 + x[2]^2
    optfunc = OptimizationFunction(quad, AutoForwardDiff())
    prob = OptimizationProblem(optfunc, zeros(2), nothing)
    nan_u0 = OptimizationProblem(optfunc, [NaN, 0.0], nothing)
    nan_lb = OptimizationProblem(
        optfunc, zeros(2), nothing;
        lb = [NaN, -1.0], ub = [1.0, 1.0]
    )
    nan_ucons = OptimizationProblem(
        OptimizationFunction(
            quad, AutoForwardDiff();
            cons = (res, x, p) -> (res[1] = x[1] + x[2])
        ),
        zeros(2), nothing; lcons = [0.0], ucons = [NaN]
    )
    @testset "$label" for (label, problem, kwargs, phrase) in (
            ("maxiters = 0", prob, (; maxiters = 0), "maxiters"),
            ("maxiters = 1.5", prob, (; maxiters = 1.5), "maxiters"),
            ("maxtime = Inf", prob, (; maxtime = Inf), "maxtime"),
            ("abstol = 0.0", prob, (; abstol = 0.0), "abstol"),
            ("reltol = NaN", prob, (; reltol = NaN), "reltol"),
            ("NaN in u0", nan_u0, (;), "u0"),
            ("NaN in lb", nan_lb, (;), "lb"),
            ("NaN in ucons", nan_ucons, (;), "ucons"),
        )
        err = try
            solve(problem, SnoptOptimizer(); kwargs...)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin(phrase, err.msg)
    end

    # solve! reads the start point that reinit! stored, not the one of the problem.
    cache = init(prob, SnoptOptimizer())
    reinit!(cache; u0 = [NaN, 0.0])
    @test_throws ArgumentError solve!(cache)
end

@testset "Integer variables are relaxed with a warning" begin
    int_prob = OptimizationProblem(
        OptimizationFunction((x, p) -> sum(abs2, x), AutoForwardDiff()),
        zeros(2), nothing; int = [true, false]
    )
    @test_logs (:warn, r"integer variables .* are relaxed") match_mode = :any begin
        init(int_prob, SnoptOptimizer())
    end
end

include("trace_contracts.jl")
include("adapter_contracts.jl")

@testset "Trace printing and storage" begin
    @test OptimizationSNOPT.snopt_show_trace(true)
    @test !OptimizationSNOPT.snopt_show_trace(OptimizationBase.DEFAULT_VERBOSE)

    raw_color_io = IOBuffer()
    color_io = IOContext(raw_color_io, :color => true)
    OptimizationSNOPT.print_trace_algorithm(color_io, SnoptOptimizer())
    color_trace = String(take!(raw_color_io))
    @test occursin("\e[32m", color_trace)
    @test occursin("Algorithm: SnoptOptimizer", color_trace)

    @testset "An snSTOP event becomes a printed row and a stored entry" begin
        io = IOBuffer()
        logger = OptimizationSNOPT.SnoptProgressLogger(
            false, nothing, true, 2, nothing, Ref(0), io;
            trace_level = OptimizationSNOPT.SnoptTraceAll(),
            store_trace = Val(true),
            lb = [0.0, 0.0],
            ub = [2.0, 2.0]
        )
        # Major iteration 2 with 5 minor iterations, objective 6, step 0.125,
        # constraint violation 0.5 and optimality error 0.25.
        event = SNOPT.SnoptStopEvent(
            7, 2, 5, 1, 1, 0,
            6.0, 6.5, 1.25, 0.125,
            0.5, 0.25, 0.5, 0.1,
            10.0, 1.0, 0.0, 6.0, 6.5,
            1, 1, 2, 3, 1, 2, 2,
            (false, true),
            [1.5, 0.5, 0.0],
            [0.0, 0.0, 1.0], [2.0, 2.0, 1.0],
            [1.5], [1.5],
            [1.0, 1.0], [3.0, 1.0],
            [0.1], [0.1],
            zeros(3), zeros(1),
            Int32[0, 0, 0]
        )
        @test logger(event)
        trace = String(take!(io))
        @test occursin(r"^\s*2\s+5\s+6\.00000000e\+00"m, trace)
        stored = OptimizationSNOPT.stored_trace(logger)
        @test stored.history[1].x == [1.5, 0.5]
        @test stored.history[1].c == [1.5]
        @test stored.history[1].constraint_violation == 0.5
        @test stored.history[1].optimality == 0.25
    end
end

# The solver tests need the licensed SNOPT library, which CI runners do not have.
if SNOPT.has_snopt()
    include("solver_tests.jl")
else
    @info "SNOPT library not found: skipping the solver tests"
end
