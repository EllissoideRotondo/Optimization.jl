using OptimizationBase, OptimizationSNOPT
using SciMLBase
using SNOPT
using Test
using ForwardDiff

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
rosenbrock_prob = OptimizationProblem(
    OptimizationFunction(rosenbrock, AutoForwardDiff()),
    zeros(2), [1.0, 100.0]
)

function captured_stdout(f)
    return mktemp() do path, io
        redirect_stdout(f, io)
        flush(io)
        read(path, String)
    end
end

@testset "Repeated solves do not reuse stale SNOPT state" begin
    for i in 1:2
        sol = solve(rosenbrock_prob, SnoptOptimizer(); maxiters = 20)
        @test sol.original.inform != 0
        @test sol.original.major_itns > 0
    end
end

function test_hs071(backend, optimizer)
    function objective(x, _)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(res, x, _)
        return res .= [
            x[1] * x[2] * x[3] * x[4],
            x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2,
        ]
    end
    prob = OptimizationProblem(
        OptimizationFunction(objective, backend; cons = constraints),
        [1.0, 5.0, 5.0, 1.0];
        sense = OptimizationBase.MinSense,
        lb = [1.0, 1.0, 1.0, 1.0],
        ub = [5.0, 5.0, 5.0, 5.0],
        lcons = [25.0, 40.0],
        ucons = [Inf, 40.0]
    )
    sol = solve(prob, optimizer)
    @test isapprox(sol.objective, 17.014017145179164; atol = 1.0e-5)
    x = [1.0, 4.742999641809297, 3.8211499817883077, 1.3794082897556983]
    @test isapprox(sol.u, x; atol = 1.0e-5)
    return
end

@testset "backends" begin
    backends = (
        AutoForwardDiff(),
        AutoSparse(AutoForwardDiff()),
    )
    for backend in backends
        @testset "$backend" begin
            test_hs071(backend, SnoptOptimizer())
        end
    end
end

include("problem_types.jl")

@testset "Cache interface (init/solve!)" begin
    cache = init(rosenbrock_prob, SnoptOptimizer())
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4

    # Evaluation counters are per-solve: a second solve! must not accumulate
    # the first solve's counts.
    sol2 = solve!(cache)
    @test SciMLBase.successful_retcode(sol2)
    @test sol2.stats.fevals == sol.stats.fevals
    @test sol2.stats.gevals == sol.stats.gevals
end

@testset "solve! after reinit! uses the new parameters" begin
    # Regression: the gradient, constraints and constraint Jacobian kept the
    # parameters of `init` after `reinit!(cache; p)`.
    # min (x1 - p1)^2 + x2^2 s.t. p2 * x1 - x2 = 0 has x1 = p1 / (1 + p2^2), x2 = p2 * x1.
    objective(x, p) = (x[1] - p[1])^2 + x[2]^2
    constraint(res, x, p) = (res[1] = p[2] * x[1] - x[2]; nothing)
    f = OptimizationFunction(objective, AutoForwardDiff(); cons = constraint)
    prob = OptimizationProblem(f, [0.5, 0.5], [1.0, 0.0]; lcons = [0.0], ucons = [0.0])
    cache = init(prob, SnoptOptimizer())
    @test solve!(cache).u ≈ [1.0, 0.0] atol = 1.0e-6
    reinit!(cache; p = [2.0, 1.0])
    @test solve!(cache).u ≈ [1.0, 1.0] atol = 1.0e-6
end

@testset "Integer variables are relaxed to a continuous solution" begin
    prob = OptimizationProblem(
        OptimizationFunction(
            (x, p) -> x[1]^2 + x[2]^2, AutoForwardDiff();
            cons = (res, x, p) -> (res[1] = x[1] + x[2])
        ),
        [2.0, 2.0], nothing;
        lb = [0.0, 0.0], ub = [5.0, 5.0], int = [true, true],
        lcons = [3.5], ucons = [Inf]
    )
    sol = solve(prob, SnoptOptimizer())
    @test sol.u ≈ [1.75, 1.75] atol = 1.0e-5
end

@testset "Common interface arguments" begin
    sol1 = solve(rosenbrock_prob, SnoptOptimizer(); abstol = 1.0e-10)
    @test sol1.u ≈ [1.0, 1.0] atol = 1.0e-8

    sol2 = solve(
        rosenbrock_prob, SnoptOptimizer(major_iterations_limit = 1000); maxiters = 3
    )
    @test sol2.retcode == SciMLBase.ReturnCode.MaxIters

    sol_time = solve(rosenbrock_prob, SnoptOptimizer(); maxtime = 10.0)
    @test SciMLBase.successful_retcode(sol_time)

    sol_trace = solve(
        rosenbrock_prob, SnoptOptimizer(); store_trace = Val(true),
        trace_level = OptimizationSNOPT.SnoptTraceMinimal(2)
    )
    trace = sol_trace.original.trace
    @test trace.history[end].status === OptimizationSNOPT.SNOPT.SnoptStatus.Optimal
    # SnoptTraceMinimal stores no copy of the point.
    @test all(entry -> entry.x === nothing, trace.history)
end

@testset "verbose = true prints the trace and the default prints nothing" begin
    verbose_output = captured_stdout(
        () -> solve(rosenbrock_prob, SnoptOptimizer(); verbose = true)
    )
    @test occursin(r"Major\s+Minor\s+Objective", verbose_output)
    all_output = captured_stdout(
        () -> solve(
            rosenbrock_prob, SnoptOptimizer(); verbose = true, trace_level = SnoptTraceAll()
        )
    )
    @test occursin("Bound viol", all_output)
    default_output = captured_stdout(() -> solve(rosenbrock_prob, SnoptOptimizer()))
    @test isempty(default_output)
end

@testset "A non-default Hessian mode solves" begin
    sol = solve(
        rosenbrock_prob,
        SnoptOptimizer(hessian = "limited_memory", major_optimality_tolerance = 1.0e-8)
    )
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-6
end

@testset "MaxSense and callback termination" begin
    linear(x, _) = x[1]
    max_prob = OptimizationProblem(
        OptimizationFunction(linear, AutoForwardDiff()),
        [0.0], nothing;
        lb = [0.0], ub = [1.0], sense = OptimizationBase.MaxSense
    )
    seen = Float64[]
    iters = Int[]
    originals = Any[]
    max_sol = solve(
        max_prob, SnoptOptimizer();
        callback = (state, objective) -> begin
            push!(seen, objective)
            push!(iters, state.iter)
            push!(originals, state.original)
            false
        end
    )
    @test max_sol.u ≈ [1.0] atol = 1.0e-8
    @test max_sol.objective ≈ 1.0 atol = 1.0e-8
    # The callback receives the user's objective, not SNOPT's negated one.
    @test seen[end] ≈ 1.0 atol = 1.0e-8
    # Regression: the callback ran on every objective evaluation with `iter = 0`.
    # It runs once per major iteration, with SNOPT's log record in `original`.
    @test iters == 0:(length(iters) - 1)
    @test all(o -> o isa SNOPT.SnoptStopEvent, originals)
    @test max_sol.stats.iterations == iters[end]

    quadratic(x, _) = (x[1] - 2)^2
    stop_prob = OptimizationProblem(
        OptimizationFunction(quadratic, AutoForwardDiff()), [0.0], nothing
    )
    stop_sol = solve(
        stop_prob, SnoptOptimizer(); callback = (_, _) -> true, store_trace = Val(true)
    )
    @test stop_sol.retcode == SciMLBase.ReturnCode.Terminated
    # The trace ends with a final entry that records the stop status.
    @test stop_sol.original.trace.history[end].status ===
        SNOPT.status_from_code(stop_sol.original.inform)
end

@testset "Unbounded problem returns a failure solution" begin
    # min x[1] with no lower bound is unbounded; SNOPT reports inform 21/22.
    # The solve must return a solution with a failure retcode, not throw.
    unbounded = OptimizationProblem(
        OptimizationFunction((x, _) -> x[1], AutoForwardDiff()), [0.0], nothing
    )
    sol = solve(unbounded, SnoptOptimizer())
    @test !SciMLBase.successful_retcode(sol)
    @test sol.original.inform in (21, 22)

    # A subsequent solve must still succeed (workspace finalized cleanly).
    ok_sol = solve(
        OptimizationProblem(
            OptimizationFunction((x, _) -> (x[1] - 3)^2, AutoForwardDiff()), [0.0], nothing
        ),
        SnoptOptimizer()
    )
    @test SciMLBase.successful_retcode(ok_sol)
    @test ok_sol.u ≈ [3.0] atol = 1.0e-6
end

@testset "Infeasible constraints return an Infeasible solution" begin
    # On the circle x^2 + y^2 = 0.5, x^3 + y^3 <= 0.5^1.5 < 1, so the two
    # equality constraints have no common point. SNOPT reports inform 13.
    function difficult_cons(res, x, _)
        res[1] = x[1]^3 + x[2]^3 - 1.0
        res[2] = x[1]^2 + x[2]^2 - 0.5
        return nothing
    end
    prob = OptimizationProblem(
        OptimizationFunction(
            (x, _) -> x[1]^4 + x[2]^4, AutoForwardDiff(); cons = difficult_cons
        ),
        [2.0, 2.0], nothing;
        lcons = [0.0, 0.0],
        ucons = [0.0, 0.0]
    )
    sol = solve(prob, SnoptOptimizer())
    @test sol.retcode == SciMLBase.ReturnCode.Infeasible
end

@testset "additional_options passthrough" begin
    opt = SnoptOptimizer(
        additional_options = Dict{String, Any}(
            "Major step limit" => 2.0,
            "Linesearch tolerance" => 0.9
        )
    )
    sol = solve(rosenbrock_prob, opt)
    @test SciMLBase.successful_retcode(sol)
end

@testset "Solves wait while another task holds the SNOPT lock" begin
    f = OptimizationFunction((x, p) -> sum(abs2, x), AutoForwardDiff())
    lock_prob = OptimizationProblem(f, [1.0])
    started = Channel{Nothing}(1)
    pending = nothing
    lock(SNOPT.SNOPT_LOCK) do
        pending = @async begin
            put!(started, nothing)
            solve(lock_prob, SnoptOptimizer(major_print_level = 0))
        end
        take!(started)
        yield()
        @test !istaskdone(pending)
    end
    @test SciMLBase.successful_retcode(fetch(pending))
end
