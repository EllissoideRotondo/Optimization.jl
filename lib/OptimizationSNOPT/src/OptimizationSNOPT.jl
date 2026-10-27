module OptimizationSNOPT

# Not re-exported: the optimization API comes from `Optimization`/`OptimizationBase`,
# which the user loads directly. This package's public surface is its own solver.
using ADTypes: ADTypes
using OptimizationBase: OptimizationBase
using Printf: Printf, @printf, @sprintf
using SNOPT: SNOPT, SnoptB, initialize, make_confun, make_dummy_confun,
    make_objfun, set_option!, snoptb!
using SciMLBase: SciMLBase, OptimizationFunction, OptimizationProblem
using SciMLLogging: SciMLLogging
using SparseArrays: SparseArrays, SparseMatrixCSC, nnz, nonzeros
using SymbolicIndexingInterface: SymbolicIndexingInterface, parameter_symbols,
    variable_symbols

export SnoptOptimizer
export SnoptTrace
export SnoptTraceAll
export SnoptTraceEntry
export SnoptTraceMinimal

"""
    SnoptOptimizer(; major_iterations_limit = 1000, minor_iterations_limit = 500,
                   major_optimality_tolerance = 1.0e-6,
                   major_feasibility_tolerance = 1.0e-6,
                   minor_feasibility_tolerance = 1.0e-6,
                   hessian = nothing,
                   additional_options = Dict{String, Any}())

Optimization.jl algorithm that solves an `OptimizationProblem` with SNOPT.

Pass it to `solve` or `init`. The problem needs the objective gradient and, if it has
constraints, the constraint Jacobian, from an automatic differentiation backend or
from the `grad` and `cons_j` keywords of `OptimizationFunction`.

# Keywords

  - `major_iterations_limit`, `minor_iterations_limit`: iteration limits, positive
    integers.
  - `major_optimality_tolerance`: tolerance on the optimality conditions.
  - `major_feasibility_tolerance`: tolerance on the nonlinear constraint violation.
  - `minor_feasibility_tolerance`: tolerance on the constraint violation in each
    subproblem.
  - `hessian`: Hessian approximation, `nothing`, `"full_memory"` or `"limited_memory"`.
    With `nothing`, no Hessian option is passed and SNOPT applies its own rule: full
    memory for at most 75 nonlinear variables, limited memory above. Pass
    `"full_memory"` on a larger problem only when limited memory needs many more
    major iterations; the dense update costs time and storage that grow with the
    square of the number of nonlinear variables. Pass `"limited_memory"` to use the
    limited-memory update at any size.
  - `additional_options`: other SNOPT options, as a dictionary from option name
    (`String` or `Symbol`) to value (integer, finite float, `String`, `Symbol`, or
    `nothing` for an option without a value). In names and in `String` and `Symbol`
    values, `_` stands for a space.

The `solve` keywords `maxiters`, `maxtime`, `abstol` and `reltol` set the SNOPT
options `Major iterations limit`, `Time limit` (CPU seconds),
`Major optimality tolerance` and `Minor feasibility tolerance`, and take precedence
over the fields.

Throws an `ArgumentError` for a value of the wrong type or range, for a `Bool` option
value, for an option in `additional_options` that has its own keyword, or for a
duplicate option name.

# Examples

```julia
opt = SnoptOptimizer(major_optimality_tolerance = 1.0e-8,
                     hessian = "limited_memory",
                     additional_options = Dict("Linesearch tolerance" => 0.9))
sol = solve(prob, opt; maxiters = 500)
```
"""
@kwdef struct SnoptOptimizer <: SciMLBase.AbstractOptimizationAlgorithm
    # Output
    major_print_level::Int = 1
    minor_print_level::Int = 0

    # Iteration limits
    major_iterations_limit::Int = 1000
    minor_iterations_limit::Int = 500

    # Tolerances
    major_optimality_tolerance::Float64 = 1.0e-6
    major_feasibility_tolerance::Float64 = 1.0e-6
    minor_feasibility_tolerance::Float64 = 1.0e-6

    hessian::Union{Nothing, String} = nothing

    additional_options::Dict{String, Any} = Dict{String, Any}()

    function SnoptOptimizer(
            major_print_level, minor_print_level,
            major_iterations_limit, minor_iterations_limit,
            major_optimality_tolerance, major_feasibility_tolerance, minor_feasibility_tolerance,
            hessian, additional_options
        )
        major_print_level = validate_nonnegative_int(:major_print_level, major_print_level)
        minor_print_level = validate_nonnegative_int(:minor_print_level, minor_print_level)
        major_iterations_limit = validate_positive_int(:major_iterations_limit, major_iterations_limit)
        minor_iterations_limit = validate_positive_int(:minor_iterations_limit, minor_iterations_limit)
        major_optimality_tolerance = validate_positive_float(
            :major_optimality_tolerance, major_optimality_tolerance
        )
        major_feasibility_tolerance = validate_positive_float(
            :major_feasibility_tolerance, major_feasibility_tolerance
        )
        minor_feasibility_tolerance = validate_positive_float(
            :minor_feasibility_tolerance, minor_feasibility_tolerance
        )
        hessian = validate_hessian_option(hessian)
        additional_options = validate_additional_options(additional_options)
        validate_snopt_optimizer_options!(
            major_print_level,
            minor_print_level,
            major_iterations_limit,
            minor_iterations_limit,
            major_optimality_tolerance,
            major_feasibility_tolerance,
            minor_feasibility_tolerance,
            hessian,
            additional_options
        )
        new(
            major_print_level, minor_print_level,
            major_iterations_limit, minor_iterations_limit,
            major_optimality_tolerance, major_feasibility_tolerance, minor_feasibility_tolerance,
            hessian, additional_options
        )
    end
end

function SciMLBase.has_init(::SnoptOptimizer)
    return true
end

SciMLBase.allowscallback(alg::SnoptOptimizer) = true
OptimizationBase.supports_sense(::SnoptOptimizer) = true

SciMLBase.supports_opt_cache_interface(alg::SnoptOptimizer) = true
SciMLBase.requiresgradient(opt::SnoptOptimizer) = true
SciMLBase.requireshessian(opt::SnoptOptimizer) = false
SciMLBase.requiresconsjac(opt::SnoptOptimizer) = true
SciMLBase.requiresconshess(opt::SnoptOptimizer) = false
SciMLBase.allowsbounds(opt::SnoptOptimizer) = true
SciMLBase.allowsconstraints(opt::SnoptOptimizer) = true

include("callback.jl")
include("cache.jl")

snopt_show_trace(verbose) = false
snopt_show_trace(verbose::Bool) = verbose
snopt_show_trace(::Val{true}) = true
snopt_show_trace(::Val{false}) = false
snopt_show_trace(::SciMLLogging.None) = false
snopt_show_trace(::SciMLLogging.AbstractVerbosityPreset) = true

function validate_integer_option(name::Symbol, value)
    value isa Integer && !(value isa Bool) ||
        throw(ArgumentError("$(name) must be an integer, got $(repr(value))"))
    return Int(value)
end

function validate_nonnegative_int(name::Symbol, value)
    value = validate_integer_option(name, value)
    value >= 0 ||
        throw(ArgumentError("$(name) must be ≥ 0, got $value"))
    return value
end

function validate_positive_int(name::Symbol, value)
    value = validate_integer_option(name, value)
    value > 0 ||
        throw(ArgumentError("$(name) must be > 0, got $value"))
    return value
end

function validate_positive_float(name::Symbol, value)
    value isa Real && !(value isa Bool) ||
        throw(ArgumentError("$(name) must be a real number, got $(repr(value))"))
    value = Float64(value)
    isfinite(value) ||
        throw(ArgumentError("$(name) must be finite, got $value"))
    value > 0 ||
        throw(ArgumentError("$(name) must be > 0, got $value"))
    return value
end

function validate_hessian_option(value)
    value === nothing && return nothing
    message = "hessian must be `nothing`, \"full_memory\" or \"limited_memory\", got $(repr(value))"
    value isa AbstractString || throw(ArgumentError(message))
    value = String(value)
    value in ("full_memory", "limited_memory") || throw(ArgumentError(message))
    return value
end

function normalize_snopt_option_key(key)
    key isa Union{AbstractString, Symbol} ||
        throw(ArgumentError("SNOPT option keys must be strings or symbols, got $(repr(key))"))
    normalized = replace(String(key), '_' => ' ')
    normalized = join(split(normalized), " ")
    isempty(normalized) &&
        throw(ArgumentError("SNOPT option keys must not be empty"))
    return uppercasefirst(lowercase(normalized))
end

function normalize_snopt_option_value(key::String, value)
    if value === nothing
        return nothing
    elseif value isa Bool
        throw(
            ArgumentError(
                "SNOPT option $(repr(key)) received Bool value $(repr(value)); " *
                    "use an integer, float, string, or `nothing` for bare options"
            )
        )
    elseif value isa Integer
        return Int(value)
    elseif value isa AbstractFloat
        value = Float64(value)
        isfinite(value) ||
            throw(ArgumentError("SNOPT option $(repr(key)) must be finite, got $value"))
        return value
    elseif value isa AbstractString || value isa Symbol
        normalized = replace(String(value), '_' => ' ')
        normalized = String(strip(normalized))
        isempty(normalized) &&
            throw(ArgumentError("SNOPT option $(repr(key)) received an empty string value"))
        return normalized
    end
    throw(
        ArgumentError(
            "SNOPT option $(repr(key)) has unsupported value $(repr(value)); " *
                "use an integer, float, string, symbol, or `nothing`"
        )
    )
end

const SNOPT_RESERVED_OPTIONS = Dict{String, String}(
    "Major print level" => "major_print_level",
    "Minor print level" => "minor_print_level",
    "Major iterations limit" => "major_iterations_limit",
    "Minor iterations limit" => "minor_iterations_limit",
    "Major optimality tolerance" => "major_optimality_tolerance",
    "Major feasibility tolerance" => "major_feasibility_tolerance",
    "Minor feasibility tolerance" => "minor_feasibility_tolerance",
    "Hessian full memory" => "hessian",
    "Hessian limited memory" => "hessian",
)

function validate_additional_options(options)
    options isa AbstractDict ||
        throw(ArgumentError("additional_options must be an AbstractDict, got $(typeof(options))"))
    normalized = Dict{String, Any}()
    for (key, value) in pairs(options)
        normalized_key = normalize_snopt_option_key(key)
        if haskey(SNOPT_RESERVED_OPTIONS, normalized_key)
            field = SNOPT_RESERVED_OPTIONS[normalized_key]
            throw(
                ArgumentError(
                    "SNOPT option $(repr(normalized_key)) is already controlled by the " *
                        "`$(field)` keyword of SnoptOptimizer; " *
                        "pass it directly to the constructor instead of via additional_options"
                )
            )
        end
        haskey(normalized, normalized_key) &&
            throw(ArgumentError("duplicate SNOPT option key after normalization: $(repr(normalized_key))"))
        normalized[normalized_key] = normalize_snopt_option_value(normalized_key, value)
    end
    return normalized
end

function hessian_option_string(hessian::String)
    return replace(hessian, "_" => " ")
end

const SNOPT_BOUND_INF = 1.0e20

function snopt_bound_value(value)
    value = Float64(value)
    value == Inf && return SNOPT_BOUND_INF
    value == -Inf && return -SNOPT_BOUND_INF
    return value
end

snopt_bound_vector(values) = snopt_bound_value.(collect(values))

function snopt_optimizer_option_pairs(
        major_print_level,
        minor_print_level,
        major_iterations_limit,
        minor_iterations_limit,
        major_optimality_tolerance,
        major_feasibility_tolerance,
        minor_feasibility_tolerance,
        hessian,
        additional_options;
        include_print_levels::Bool = true
    )
    options = Tuple{String, Any}[]
    if include_print_levels
        push!(options, ("Major print level", major_print_level))
        push!(options, ("Minor print level", minor_print_level))
    end
    push!(options, ("Major iterations limit", major_iterations_limit))
    push!(options, ("Minor iterations limit", minor_iterations_limit))
    push!(options, ("Major optimality tolerance", major_optimality_tolerance))
    push!(options, ("Major feasibility tolerance", major_feasibility_tolerance))
    push!(options, ("Minor feasibility tolerance", minor_feasibility_tolerance))
    # Without a Hessian option, SNOPT chooses full or limited memory from the number
    # of nonlinear variables.
    hessian === nothing ||
        push!(options, ("Hessian $(hessian_option_string(hessian))", nothing))
    for (key, value) in additional_options
        push!(options, (key, value))
    end
    return options
end

snopt_optimizer_option_pairs(opt::SnoptOptimizer; include_print_levels::Bool = true) =
    snopt_optimizer_option_pairs(
    opt.major_print_level,
    opt.minor_print_level,
    opt.major_iterations_limit,
    opt.minor_iterations_limit,
    opt.major_optimality_tolerance,
    opt.major_feasibility_tolerance,
    opt.minor_feasibility_tolerance,
    opt.hessian,
    opt.additional_options;
    include_print_levels
)

function snopt_option_label(key::String, value)
    value === nothing && return repr(key)
    return "$(repr(key)) => $(repr(value))"
end

function apply_snopt_option!(ws, key::String, value)
    errors = nothing
    if value === nothing
        errors = set_option!(ws, key)
    elseif value isa Int
        errors = set_option!(ws, key, value)
    elseif value isa Float64
        errors = set_option!(ws, key, value)
    elseif value isa String
        errors = set_option!(ws, string(key, " ", value))
    else
        throw(ArgumentError("unsupported normalized SNOPT option value $(repr(value))"))
    end
    if errors isa Integer && errors != 0
        msg = value === nothing ?
            "SNOPT: Couldn't set option '$(key)'" :
            "SNOPT: Couldn't set option '$(key)' to value '$(value)'"
        throw(ArgumentError(msg))
    end
    return ws
end

function validate_snopt_option_pairs_with_library!(options)
    SNOPT.has_snopt() || return nothing
    # Note: this creates (and closes) a temporary SNOPT workspace, which — by
    # SNOPT's single-active-workspace rule — finalizes any manually managed
    # SNOPT.jl workspace that is currently open in the process.
    lock(SNOPT.SNOPT_LOCK) do
        ws = initialize("", "", 1000, 1000)
        try
            redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    for (key, value) in options
                        apply_snopt_option!(ws, key, value)
                    end
                end
            end
        finally
            close(ws)
        end
    end
    return nothing
end

function validate_snopt_optimizer_options!(
        major_print_level,
        minor_print_level,
        major_iterations_limit,
        minor_iterations_limit,
        major_optimality_tolerance,
        major_feasibility_tolerance,
        minor_feasibility_tolerance,
        hessian,
        additional_options
    )
    options = snopt_optimizer_option_pairs(
        major_print_level,
        minor_print_level,
        major_iterations_limit,
        minor_iterations_limit,
        major_optimality_tolerance,
        major_feasibility_tolerance,
        minor_feasibility_tolerance,
        hessian,
        additional_options
    )
    validate_snopt_option_pairs_with_library!(options)
    return nothing
end

function configure_snopt_options!(
        ws,
        opt::SnoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing
    )
    for (key, value) in snopt_optimizer_option_pairs(opt)
        apply_snopt_option!(ws, key, value)
    end

    !isnothing(maxiters) && set_option!(ws, "Major iterations limit", Int(maxiters))
    !isnothing(maxtime)  && set_option!(ws, "Time limit", Float64(maxtime))
    !isnothing(abstol)   && set_option!(ws, "Major optimality tolerance", Float64(abstol))
    !isnothing(reltol)   && set_option!(ws, "Minor feasibility tolerance", Float64(reltol))

    set_option!(ws, "Solution = No")

    return ws
end

function snopt_workspace_lengths(
        n::Int,
        m_eff::Int,
        nc::Int,
        J::SparseMatrixCSC,
        opt::SnoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing
    )
    neJ = nnz(J)
    mem_ws = initialize("", "", 1000, 1000)
    return try
        configure_snopt_options!(mem_ws, opt; maxiters, maxtime, abstol, reltol)
        negCon = nc > 0 ? neJ : 0
        nnCon = nc
        nnJac = nc > 0 ? n : 0
        nnObj = n
        memory = SNOPT.snmemb(mem_ws, m_eff, n, neJ, negCon, nnCon, nnObj, nnJac)
        if memory.info == 100 || memory.info == 104
            return memory.miniw, memory.minrw
        end
        error("SNOPT memory estimator failed with info code $(memory.info)")
    finally
        close(mem_ws)
    end
end

function map_optimizer_args(
        cache,
        opt::SnoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose = false,
        trace_level = SnoptTraceMinimal(),
        store_trace = Val(false),
        progress::Bool = false,
        callback = nothing
    )
    n = cache.n
    nc = cache.num_cons

    m_eff = nc > 0 ? nc : 1

    user_lb = isnothing(cache.lb) ? fill(-Inf, n) : Vector{Float64}(cache.lb)
    user_ub = isnothing(cache.ub) ? fill(Inf, n) : Vector{Float64}(cache.ub)
    user_lcons = nc > 0 ? Vector{Float64}(cache.lcons) : Float64[]
    user_ucons = nc > 0 ? Vector{Float64}(cache.ucons) : Float64[]
    for (name, values) in (
            ("lb", user_lb), ("ub", user_ub),
            ("lcons", user_lcons), ("ucons", user_ucons),
        )
        any(isnan, values) &&
            throw(ArgumentError("$name must not contain NaN"))
    end
    lb = snopt_bound_vector(user_lb)
    ub = snopt_bound_vector(user_ub)
    bl = vcat(lb, nc > 0 ? snopt_bound_vector(user_lcons) : [-SNOPT_BOUND_INF])
    bu = vcat(ub, nc > 0 ? snopt_bound_vector(user_ucons) : [SNOPT_BOUND_INF])
    x_ext = zeros(Float64, n + m_eff)
    hs = zeros(Int32, n + m_eff)

    # Build Jacobian with Int32 indices required by SNOPT.
    # When nc=0, SNOPT requires m>=1; we add a dummy linear constraint row
    # with a single zero coefficient at (1,1) and bounds [-Inf, Inf].
    J = if nc == 0
        SparseMatrixCSC{Float64, Int32}(
            1, n,
            Int32.(vcat(1, fill(2, n))),
            Int32[1], Float64[0.0]
        )
    elseif cache.J isa SparseMatrixCSC
        SparseMatrixCSC{Float64, Int32}(
            nc, n,
            Int32.(cache.J.colptr), Int32.(cache.J.rowval),
            zeros(Float64, nnz(cache.J))
        )
    else
        rowval = repeat(Int32.(1:nc), n)
        colptr = Int32.(range(1; step = nc, length = n + 1))
        SparseMatrixCSC(nc, n, colptr, rowval, zeros(Float64, nc * n))
    end

    leniw, lenrw = snopt_workspace_lengths(
        n, m_eff, nc, J, opt; maxiters, maxtime, abstol, reltol
    )

    show_output = snopt_show_trace(verbose)
    # Keep SNOPT's own print/summary files disabled. Verbose output is produced
    # synchronously by the Julia callback below, so Windows and single-threaded
    # runs do not buffer the whole SNOPT summary until finalization.
    ws = initialize("", "", leniw, lenrw)
    try
        logger = SnoptProgressLogger(
            progress, callback, show_output, n, maxiters, cache.iterations;
            trace_level, store_trace, lb = user_lb, ub = user_ub,
            algorithm = opt,
            sense = cache.sense,
            ws_rw = ws.rw
        )

        objfun = make_objfun(
            x -> eval_objective(cache, x),
            (g, x) -> eval_objective_gradient(cache, g, x),
            ws.iw
        )

        active_confun = if nc > 0
            make_confun(
                (c, x) -> eval_constraint(cache, c, x),
                (jnzval, x) -> eval_constraint_jacobian(cache, jnzval, x),
                J,
                ws.iw
            )
        else
            make_dummy_confun()
        end

        configure_snopt_options!(ws, opt; maxiters, maxtime, abstol, reltol)

        return SnoptB(
                ws, n, nc, m_eff, n, x_ext, bl, bu, hs, J,
                0.0, 0, Float64[], objfun, active_confun
            ), logger
    catch
        # Do not leave a half-configured workspace alive until the GC runs
        # (e.g. when SNOPT rejects an option that was valid at construction
        # time but not for this solve, such as "Time limit" on older builds).
        close(ws)
        rethrow()
    end
end

function check_and_convert_maxiters(maxiters::Nothing)
    return nothing
end

function check_and_convert_maxiters(maxiters)
    maxiters isa Integer && !(maxiters isa Bool) ||
        throw(ArgumentError("maxiters must be an integer, got $(repr(maxiters))"))
    maxiters > 0 ||
        throw(ArgumentError("maxiters must be > 0, got $maxiters"))
    return Int(maxiters)
end

function check_and_convert_maxtime(maxtime::Nothing)
    return nothing
end

function check_and_convert_maxtime(maxtime)
    maxtime isa Real && !(maxtime isa Bool) ||
        throw(ArgumentError("maxtime must be a real number, got $(repr(maxtime))"))
    maxtime = Float64(maxtime)
    isfinite(maxtime) ||
        throw(ArgumentError("maxtime must be finite, got $maxtime"))
    maxtime > 0 ||
        throw(ArgumentError("maxtime must be > 0, got $maxtime"))
    return maxtime
end

function check_and_convert_tolerance(name::Symbol, tolerance::Nothing)
    return nothing
end

function check_and_convert_tolerance(name::Symbol, tolerance)
    tolerance isa Real && !(tolerance isa Bool) ||
        throw(ArgumentError("$(name) must be a real number, got $(repr(tolerance))"))
    tolerance = Float64(tolerance)
    isfinite(tolerance) ||
        throw(ArgumentError("$(name) must be finite, got $tolerance"))
    tolerance > 0 ||
        throw(ArgumentError("$(name) must be > 0, got $tolerance"))
    return tolerance
end

function map_retcode(inform::Int)
    if inform in (1, 2, 3)
        return SciMLBase.ReturnCode.Success
    elseif inform in (5, 6, 11, 12, 13, 14, 15, 16)
        # 5 and 6 are elastic-mode results, in which the constraints may be violated.
        return SciMLBase.ReturnCode.Infeasible
    elseif inform in (21, 22)
        # Unbounded objective. SciMLBase has no dedicated "unbounded" code, so
        # report a generic failure (the SNOPT inform is preserved in
        # sol.original.inform). Note: ReturnCode has no `DivergeFailed`.
        return SciMLBase.ReturnCode.Failure
    elseif inform in (31, 32)
        return SciMLBase.ReturnCode.MaxIters
    elseif inform == 34
        return SciMLBase.ReturnCode.MaxTime
        # 33 ("superbasics limit is too small") is a sizing failure, not an
        # iteration limit, and falls through to Failure below.
    elseif inform in (71, 72, 73, 74)
        return SciMLBase.ReturnCode.Terminated
    else
        return SciMLBase.ReturnCode.Failure
    end
end

function run_snopt_attempt(
        cache::SnoptCache,
        maxiters::Union{Int, Nothing},
        maxtime::Union{Float64, Nothing},
        abstol::Union{Float64, Nothing},
        reltol::Union{Float64, Nothing}
    )
    opt_setup, logger = map_optimizer_args(
        cache,
        cache.opt;
        abstol = abstol,
        reltol = reltol,
        maxiters = maxiters,
        maxtime = maxtime,
        verbose = cache.solver_args.show_trace,
        trace_level = cache.solver_args.trace_level,
        store_trace = cache.solver_args.store_trace,
        progress = cache.progress,
        callback = cache.callback
    )

    opt_setup.x[1:cache.n] .= cache.reinit_cache.u0

    try
        if uses_snstop(logger)
            snoptb!(opt_setup; snstop = logger)
        else
            snoptb!(opt_setup)
        end
    catch
        close(opt_setup.ws)
        rethrow()
    end

    return opt_setup, logger
end

function snopt_returned_without_evaluating(opt_setup)
    return opt_setup.status == 0 &&
        opt_setup.ws.major_itns == 0 &&
        opt_setup.ws.iterations == 0
end

function SciMLBase.__solve(cache::SnoptCache)
    maxiters = check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = check_and_convert_maxtime(cache.solver_args.maxtime)
    abstol = check_and_convert_tolerance(:abstol, cache.solver_args.abstol)
    reltol = check_and_convert_tolerance(:reltol, cache.solver_args.reltol)
    all(isfinite, cache.reinit_cache.u0) ||
        throw(ArgumentError("u0 must contain only finite values"))

    # Evaluation counters are per-solve statistics; reset them so repeated
    # solve!(cache) calls do not accumulate across solves.
    cache.f_calls = 0
    cache.f_grad_calls = 0
    cache.iterations[] = 0

    start_time = time()
    # Hold SNOPT's process lock across the attempts and the result reads, because
    # Fortran permits one active workspace. Solves run sequentially; use separate
    # processes for parallel solves.
    opt_setup, logger, minimizer, lambda = lock(SNOPT.SNOPT_LOCK) do
        opt_setup, logger = run_snopt_attempt(cache, maxiters, maxtime, abstol, reltol)
        if snopt_returned_without_evaluating(opt_setup)
            close(opt_setup.ws)
            @warn "SNOPT returned inform=0 without evaluating the problem; retrying once with a fresh workspace"
            opt_setup, logger = run_snopt_attempt(cache, maxiters, maxtime, abstol, reltol)
        end

        # Read results before finalization in case f_snend touches the
        # workspace arrays.
        minimizer = opt_setup.ws.x[1:cache.n]
        lambda = copy(opt_setup.lambda[1:(cache.n + cache.num_cons)])

        # Call f_snend immediately rather than relying on the GC finalizer.
        # SNOPT7 has global Fortran state; without this, a second solve in the
        # same process sees stale state and returns instantly with inform=0.
        close(opt_setup.ws)
        opt_setup, logger, minimizer, lambda
    end

    opt_ret = map_retcode(opt_setup.status)
    internal_minimum = opt_setup.obj_val
    minimum = cache.sense === OptimizationBase.MaxSense ? -internal_minimum : internal_minimum
    iterations = opt_setup.ws.iterations
    major_itns = opt_setup.ws.major_itns
    num_inf = opt_setup.ws.num_inf
    sum_inf = opt_setup.ws.sum_inf
    trace_objective = minimum
    if !snopt_returned_without_evaluating(opt_setup)
        finish_trace!(
            logger, opt_setup.status, trace_objective;
            major_iter = major_itns, minor_iter = iterations
        )
    end
    trace = stored_trace(logger)

    if cache.progress
        Base.@logmsg(Base.LogLevel(-1), "", progress = 1)
    end

    cache.iterations[] = major_itns
    stats = OptimizationBase.OptimizationStats(;
        time = time() - start_time,
        iterations = major_itns,
        fevals = cache.f_calls,
        gevals = cache.f_grad_calls
    )

    return SciMLBase.build_solution(
        cache,
        cache.opt,
        minimizer,
        minimum;
        original = (inform = opt_setup.status, lambda, iterations, major_itns, num_inf, sum_inf, trace),
        retcode = opt_ret,
        stats = stats
    )
end

function SciMLBase.__init(
        prob::OptimizationProblem,
        opt::SnoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress::Bool = false,
        show_trace = nothing,
        trace_level = SnoptTraceMinimal(),
        store_trace = Val(false),
        verbose = OptimizationBase.DEFAULT_VERBOSE,
        kwargs...
    )
    final_verbose = isnothing(show_trace) ? verbose : show_trace
    return SnoptCache(
        prob, opt; maxiters, maxtime, abstol, reltol, progress,
        verbose = final_verbose, trace_level, store_trace, kwargs...
    )
end


end # module OptimizationSNOPT
