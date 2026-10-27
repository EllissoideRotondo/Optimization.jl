# OptimizationSNOPT.jl

`OptimizationSNOPT.jl` is a wrapper package that integrates
[`SNOPT.jl`](https://ellissoiderotondo.github.io/SNOPT.jl/stable/) with the
[`Optimization.jl`](https://github.com/SciML/Optimization.jl) ecosystem, so that the
SNOPT solver can be used through Optimization.jl's common interface.

[SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) is a commercial sequential
quadratic programming (SQP) solver for large, sparse, constrained nonlinear problems.
SNOPT solves problems of the form

```math
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{s.t.} \quad & l_c \leq c(x) \leq u_c \\
& l_x \leq x \leq u_x
\end{aligned}
```

where ``f`` is the objective, ``c`` are the constraint functions, ``l_c`` and ``u_c``
are the lower and upper bounds on the constraint values (`lcons` and `ucons`), and
``l_x`` and ``u_x`` are the bounds on the variables (`lb` and `ub`). Equal lower and
upper bounds give an equality constraint.

## Installation: OptimizationSNOPT.jl

To use this package, install the OptimizationSNOPT package:

```julia
import Pkg;
Pkg.add("OptimizationSNOPT");
```

The package does not include SNOPT. It needs a licensed SNOPT 7.7 library,
`libsnopt7`, built with the [snopt-interface](https://github.com/snopt/snopt-interface)
C functions. Before starting Julia, set `SNOPTDIR` to the directory that contains the
library:

```bash
export SNOPTDIR=/path/to/snopt/lib
```

`SNOPT.has_snopt()`, from SNOPT.jl, returns `true` when the library is loaded. The
[SNOPT.jl installation guide](https://ellissoiderotondo.github.io/SNOPT.jl/stable/installation/)
gives the steps for each operating system.

## Methods

```@docs
OptimizationSNOPT.SnoptOptimizer
```

### Algorithm Requirements

`SnoptOptimizer` requires:

  - The gradient of the objective.
  - The constraint Jacobian, for a constrained problem.

Choose an automatic differentiation backend in `OptimizationFunction`, for example
`AutoForwardDiff()`, and load its package. SNOPT builds its own approximation of the
Hessian of the Lagrangian, so no second derivatives are computed.

The algorithm supports:

  - Bounds on the variables, with `lb` and `ub` in the `OptimizationProblem`.
  - Nonlinear equality and inequality constraints, with `lcons` and `ucons`.
  - Sparse constraint Jacobians, from a sparse backend such as
    `AutoSparse(AutoForwardDiff())` or from `cons_jac_prototype`.

### Basic Usage

```julia
using Optimization, OptimizationSNOPT

# Create the optimizer with the default settings
opt = SnoptOptimizer()

# Or set SNOPT options
opt = SnoptOptimizer(
    major_optimality_tolerance = 1.0e-8,
    hessian = "limited_memory"
)

# Solve the problem
sol = solve(prob, opt)
```

## Options and Parameters

### Common Interface Options

The following keyword arguments of `solve` follow the common Optimization.jl
interface:

  - `maxiters`: maximum number of major iterations (sets SNOPT's
    `Major iterations limit`).
  - `maxtime`: time limit in CPU seconds (sets `Time limit`).
  - `abstol`: tolerance on the optimality conditions (sets
    `Major optimality tolerance`).
  - `reltol`: tolerance on the constraint violation in each QP subproblem (sets
    `Minor feasibility tolerance`).
  - `verbose`: `true` prints the progress table, by default one line per major
    iteration. Without `verbose`, nothing is printed.
  - `callback`: function called once per major iteration; see
    [Progress Output and Callbacks](@ref snopt-progress).

### SnoptOptimizer Constructor Options

The main SNOPT options are fields of `SnoptOptimizer`.

#### Iteration Limits

  - `major_iterations_limit::Int = 1000`: maximum number of major iterations.
  - `minor_iterations_limit::Int = 500`: limit on the minor iterations in the
    optimality phase of each QP subproblem.

#### Tolerances

  - `major_optimality_tolerance::Float64 = 1e-6`: tolerance on the optimality
    conditions.
  - `major_feasibility_tolerance::Float64 = 1e-6`: tolerance on the violation of the
    nonlinear constraints.
  - `minor_feasibility_tolerance::Float64 = 1e-6`: tolerance on the violation of the
    bounds and the linear constraints in each subproblem.

#### Hessian Options

  - `hessian = nothing`: approximation of the Hessian of the Lagrangian.
      - `nothing`: SNOPT chooses. It uses a dense quasi-Newton update for at most 75
        nonlinear variables, and a limited-memory update above.
      - `"full_memory"`: dense update. Its cost grows with the square of the number
        of nonlinear variables.
      - `"limited_memory"`: limited-memory update, for any problem size.

### Additional Options Dictionary

Give other SNOPT options in the `additional_options` dictionary, with the option
name as in the SNOPT user guide:

```julia
opt = SnoptOptimizer(
    additional_options = Dict(
        "Verify level" => 3,
        "Major step limit" => 0.5,
        "Scale option" => 1
    )
)
```

The [SNOPT option reference](https://ccom.ucsd.edu/~optimizers/docs/snopt/options.html)
lists all options. An option that is a field of `SnoptOptimizer`, such as
`Major iterations limit`, must be set with its field; `additional_options` gives an
`ArgumentError` for it.

### Option Priority

Options follow this priority, from highest to lowest:

 1. The keywords `maxiters`, `maxtime`, `abstol` and `reltol` of `solve`.
 2. The fields of `SnoptOptimizer` and the entries of `additional_options`.
 3. The SNOPT defaults.

```julia
opt = SnoptOptimizer(
    major_iterations_limit = 200,              # field
    additional_options = Dict("Time limit" => 60.0)
)

sol = solve(prob, opt;
    maxiters = 500,   # replaces major_iterations_limit = 200
    maxtime = 10.0    # replaces "Time limit" => 60.0
)
```

## Examples

### Basic Unconstrained Optimization

```julia
using Optimization, OptimizationSNOPT, ForwardDiff

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
f = OptimizationFunction(rosenbrock, AutoForwardDiff())
prob = OptimizationProblem(f, [0.0, 0.0], [1.0, 100.0])
sol = solve(prob, SnoptOptimizer())
```

### Box-Constrained Optimization

```julia
using Optimization, OptimizationSNOPT, ForwardDiff

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
f = OptimizationFunction(rosenbrock, AutoForwardDiff())
prob = OptimizationProblem(f, [0.0, 0.0], [1.0, 100.0];
    lb = [-1.0, -1.0], ub = [0.8, 0.8])
sol = solve(prob, SnoptOptimizer())
```

The solution lies on the bound `x[1] = 0.8`.

### Nonlinear Constrained Optimization

Problem 71 of the Hock–Schittkowski test set has bounds, one inequality and one
equality constraint:

```julia
using Optimization, OptimizationSNOPT, ForwardDiff

objective(x, p) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
function constraints!(res, x, p)
    res[1] = x[1] * x[2] * x[3] * x[4]   # inequality: res[1] >= 25
    res[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2   # equality: res[2] == 40
    return nothing
end

f = OptimizationFunction(objective, AutoSparse(AutoForwardDiff()); cons = constraints!)
prob = OptimizationProblem(f, [1.0, 5.0, 5.0, 1.0];
    lb = fill(1.0, 4), ub = fill(5.0, 4),
    lcons = [25.0, 40.0], ucons = [Inf, 40.0])
sol = solve(prob, SnoptOptimizer(major_optimality_tolerance = 1.0e-8))
```

The optimal objective value is approximately 17.014.

### [Progress Output and Callbacks](@id snopt-progress)

`verbose = true` prints one line per major iteration. `store_trace = Val(true)` stores
the iterations in `sol.original.trace`, at the detail that `trace_level` selects.
`SnoptTraceAll()` also stores the point `x` and the constraint values `c` of each
iteration:

```julia
sol = solve(prob, SnoptOptimizer(); verbose = true,
    store_trace = Val(true), trace_level = SnoptTraceAll())
sol.original.trace.history[end]
```

A `callback` runs once per major iteration. Its `state.iter` is the major iteration
number, and `state.original` is a
[`SNOPT.SnoptStopEvent`](https://ellissoiderotondo.github.io/SNOPT.jl/stable/api/monitoring/#SNOPT.SnoptStopEvent)
with the values that SNOPT reports for that iteration. A callback that returns `true`
stops the solve:

```julia
stop_early(state, objective) = state.iter >= 3
sol = solve(prob, SnoptOptimizer(); callback = stop_early)
sol.retcode   # ReturnCode.Terminated
```

```@docs
OptimizationSNOPT.SnoptTraceMinimal
OptimizationSNOPT.SnoptTraceAll
OptimizationSNOPT.SnoptTrace
OptimizationSNOPT.SnoptTraceEntry
```

## Tips and Best Practices

 1. **Sparse Jacobians**: for a problem with many variables or constraints, use a
    sparse backend such as `AutoSparse(AutoForwardDiff())`. SNOPT then stores and
    factorizes only the entries of the Jacobian that can be nonzero.

 2. **Initial Points**: SNOPT is a local optimizer. The solution it finds depends on
    the starting point `u0`, so give a good initial guess when possible.

 3. **Bounds**: give bounds on the variables where they are known. The SNOPT user
    guide recommends bounds to keep SNOPT from evaluating the functions at points
    where they are not defined, for example `log(x)` with `x <= 0`.

 4. **Hessian Approximation**: keep `hessian = nothing` first. If a problem with more
    than 75 nonlinear variables needs many major iterations, try
    `hessian = "full_memory"`, as long as its time and memory are acceptable.

 5. **Derivative Checks**: with hand-written `grad` or `cons_j` functions, set
    `"Verify level" => 3` in `additional_options` while the problem is in
    development. Incorrect derivatives then give the SNOPT status code 51 or 52 in
    `sol.original.inform`.

 6. **Rapidly Changing Functions**: if a function overflows or changes rapidly,
    a smaller `"Major step limit"` (default 2.0), for example `0.1`, limits the
    change in `x` in each line search.

 7. **Infeasible Problems**: `ReturnCode.Infeasible` means that SNOPT could not
    satisfy the constraints. For the SNOPT status codes 5 and 6 in
    `sol.original.inform`, SNOPT minimized the constraint violation instead and
    returned a point that violates the constraints. Check the constraint bounds and
    the starting point.

 8. **Status Codes**: `sol.original.inform` holds the SNOPT status code, which is
    more specific than `sol.retcode`. The
    [SNOPT 7.7 user guide](https://ccom.ucsd.edu/~optimizers/static/pdfs/sndoc7.pdf)
    describes each code.

## Limits

  - The SNOPT library keeps its state in memory shared by the whole process, so a
    Julia process runs one SNOPT solve at a time. A solve started from another task
    waits until the running solve ends. Use several processes for parallel solves.
  - Integer variables are relaxed to continuous variables, with a warning.
  - The constraint Jacobian is dense unless the backend is sparse or
    `cons_jac_prototype` gives its sparsity pattern.

## References

  - [SNOPT 7.7 user guide](https://ccom.ucsd.edu/~optimizers/static/pdfs/sndoc7.pdf)
  - [SNOPT option reference](https://ccom.ucsd.edu/~optimizers/docs/snopt/options.html)
  - [SNOPT.jl manual](https://ellissoiderotondo.github.io/SNOPT.jl/stable/)
