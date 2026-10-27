using OptimizationBase, OptimizationSNOPT
using Test
using LinearAlgebra
using Random
using ADTypes, ForwardDiff

@testset "Sparse Jacobian of a banded problem" begin
    # Discretized control problem: minimize control effort subject to linear
    # dynamics. The 21 x 39 constraint Jacobian is banded, so its CSC order
    # differs from the dense column-major order of a structurally full matrix.
    N = 20
    dt = 0.1

    function control_objective(z, _)
        u_start = N + 1
        return sum(z[i]^2 for i in u_start:length(z))
    end

    function dynamics_constraints(res, z, _)
        for i in 1:(N - 1)
            res[i] = z[i + 1] - z[i] - dt * z[N + i]
        end
        res[N] = z[1] - 0.0
        res[N + 1] = z[N] - 1.0
        return nothing
    end

    n_vars = N + (N - 1)
    n_cons = N + 1

    optfunc = OptimizationFunction(
        control_objective, AutoSparse(AutoForwardDiff());
        cons = dynamics_constraints
    )
    prob = OptimizationProblem(
        optfunc, zeros(n_vars);
        lcons = zeros(n_cons),
        ucons = zeros(n_cons)
    )

    sol = solve(prob, SnoptOptimizer())

    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 0.0 atol = 1.0e-5
    @test sol.u[N] ≈ 1.0 atol = 1.0e-5
end

@testset "Unconstrained problem with n = 100 variables" begin
    # More than 75 variables: SNOPT selects the limited-memory Hessian, and the
    # workspace is sized for the larger problem.
    rng = Xoshiro(20261026)
    n = 100
    A = randn(rng, n, n)
    Q = A' * A + I
    b = randn(rng, n)

    function large_quadratic(x, _)
        return 0.5 * dot(x, Q * x) - dot(b, x)
    end

    optfunc = OptimizationFunction(large_quadratic, AutoForwardDiff())
    prob = OptimizationProblem(optfunc, randn(rng, n))

    sol = solve(prob, SnoptOptimizer(); maxiters = 1000)

    @test SciMLBase.successful_retcode(sol)
    @test norm(Q * sol.u - b) < 1.0e-4
end
