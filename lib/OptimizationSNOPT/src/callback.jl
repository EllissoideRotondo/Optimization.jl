struct SnoptTraceLevel
    trace_mode::Symbol
    print_frequency::Int
    store_frequency::Int
end

const SNOPT_DUAL_INFEASIBILITY_INDEX = 430
const TRACE_HEADER_REPEAT_INTERVAL = 25

function validate_trace_frequency(name::Symbol, value)
    value isa Integer && !(value isa Bool) ||
        throw(ArgumentError("$(name) must be an integer, got $(repr(value))"))
    value > 0 ||
        throw(ArgumentError("$(name) must be positive, got $value"))
    return Int(value)
end

"""
    SnoptTraceMinimal(; print_frequency = 1, store_frequency = 1)
    SnoptTraceMinimal(frequency::Integer)

Trace level that prints and stores the scalar progress values of each major iteration.

Pass it as the `trace_level` keyword of `solve`. A row is printed every
`print_frequency` major iterations and stored every `store_frequency` major
iterations; the second form sets both to `frequency`. Both must be positive integers.
Stored rows have `x` and `c` set to `nothing`.
"""
function SnoptTraceMinimal(; print_frequency = 1, store_frequency = 1)
    print_frequency = validate_trace_frequency(:print_frequency, print_frequency)
    store_frequency = validate_trace_frequency(:store_frequency, store_frequency)
    return SnoptTraceLevel(:minimal, print_frequency, store_frequency)
end

"""
    SnoptTraceAll(; print_frequency = 1, store_frequency = 1)
    SnoptTraceAll(frequency::Integer)

Trace level that also stores the point and the constraint values of each major
iteration.

The keywords are the same as for [`SnoptTraceMinimal`](@ref). Each stored row holds
copies of `x` and `c`, and the printed table has an extra column with the largest
bound violation.
"""
function SnoptTraceAll(; print_frequency = 1, store_frequency = 1)
    print_frequency = validate_trace_frequency(:print_frequency, print_frequency)
    store_frequency = validate_trace_frequency(:store_frequency, store_frequency)
    return SnoptTraceLevel(:all, print_frequency, store_frequency)
end

SnoptTraceMinimal(freq::Integer) = SnoptTraceMinimal(;
    print_frequency = freq, store_frequency = freq
)
SnoptTraceAll(freq::Integer) = SnoptTraceAll(;
    print_frequency = freq, store_frequency = freq
)

"""
    SnoptTraceEntry

One row of a [`SnoptTrace`](@ref).

A value that is not available is `NaN`, or `nothing` for the vectors and the status.

# Fields

  - `iteration`: count of major-iteration records since the start of the solve,
    including those not stored; `-1` for the final row.
  - `major_iter`, `minor_iter`: number of the major iteration, and number of minor
    iterations in it.
  - `objective`: objective value of the original problem.
  - `step_norm`: length of the last step.
  - `constraint_violation`: constraint violation, as SNOPT measures it.
  - `optimality`: error in the optimality conditions, as SNOPT measures it.
  - `bound_violation`: largest violation of a variable bound.
  - `x`: copy of the point, with [`SnoptTraceAll`](@ref) only.
  - `c`: copy of the constraint values, with [`SnoptTraceAll`](@ref) only.
  - `status`: outcome of the solve, a `SNOPT.SnoptStatus` value, in the final row only.
"""
struct SnoptTraceEntry
    iteration::Int
    major_iter::Int
    minor_iter::Int
    objective::Float64
    step_norm::Float64
    constraint_violation::Float64
    optimality::Float64
    bound_violation::Float64
    x::Union{Nothing, Vector{Float64}}
    c::Union{Nothing, Vector{Float64}}
    status::Union{Nothing, SNOPT.SnoptStatus.T}
end

function Base.show(io::IO, ::MIME"text/plain", entry::SnoptTraceEntry)
    print(io, "SnoptTraceEntry(major_iter = ", entry.major_iter)
    print(io, ", minor_iter = ", entry.minor_iter)
    print(io, ", objective = ", entry.objective)
    print(io, ", constraint_violation = ", entry.constraint_violation)
    print(io, ", optimality = ", entry.optimality)
    print(io, ", step_norm = ", entry.step_norm)
    entry.x === nothing || print(io, ", x = ", entry.x)
    entry.status === nothing || print(io, ", status = ", entry.status)
    print(io, ")")
    return nothing
end

"""
    SnoptTrace

Rows stored during a solve, available as `sol.original.trace` after
`solve(prob, SnoptOptimizer(); store_trace = Val(true))`.

# Fields

  - `history`: vector of [`SnoptTraceEntry`](@ref), one per stored row.
  - `trace_level`: the trace level of the solve.
"""
struct SnoptTrace
    history::Vector{SnoptTraceEntry}
    trace_level::SnoptTraceLevel
end

function Base.show(io::IO, ::MIME"text/plain", trace::SnoptTrace)
    isempty(trace.history) && return print(io, "Tracing Disabled")
    print_trace_header(io, trace.trace_level)
    for entry in trace.history
        print_trace_entry(io, entry, trace.trace_level)
    end
    return nothing
end

function trace_level_mode(level::SnoptTraceLevel)
    return level.trace_mode
end

function trace_level_mode(level)
    mode = hasproperty(level, :trace_mode) ? getproperty(level, :trace_mode) : Val(:minimal)
    if mode isa Val{:all} || mode === :all
        return :all
    end
    return :minimal
end

function trace_print_frequency(level)
    return hasproperty(level, :print_frequency) ? Int(getproperty(level, :print_frequency)) : 1
end

function trace_store_frequency(level)
    return hasproperty(level, :store_frequency) ? Int(getproperty(level, :store_frequency)) : 1
end

function normalize_trace_level(level)
    mode = trace_level_mode(level)
    print_frequency = trace_print_frequency(level)
    store_frequency = trace_store_frequency(level)
    mode === :all && return SnoptTraceAll(; print_frequency, store_frequency)
    return SnoptTraceMinimal(; print_frequency, store_frequency)
end

snopt_store_trace(store_trace) = false
snopt_store_trace(store_trace::Bool) = store_trace
snopt_store_trace(::Val{true}) = true
snopt_store_trace(::Val{false}) = false

# SNOPT calls the logger through `snstop` once per major iteration. It prints and
# stores the trace, reports progress, and calls the user callback. `snstop`, not
# `snlog`, because SNOPT 7.7 ignores a stop request from `snlog`.
mutable struct SnoptProgressLogger{C, I, A, S}
    progress::Bool
    callback::C
    show_trace::Bool
    store_trace::Bool
    trace_io::I
    trace_level::SnoptTraceLevel
    algorithm::A
    sense::S
    n::Int
    maxiters::Union{Nothing, Int}
    iterations::Ref{Int}
    trace_evals::Ref{Int}
    trace_header_printed::Ref{Bool}
    final_trace_printed::Ref{Bool}
    u::Vector{Float64}
    has_u::Ref{Bool}
    lb::Vector{Float64}
    ub::Vector{Float64}
    last_constraint_violation::Ref{Float64}
    last_optimality::Ref{Float64}
    ws_rw::Vector{Float64}
    rows_printed::Ref{Int}
    trace_history::Vector{SnoptTraceEntry}
end

function SnoptProgressLogger(
        progress,
        callback,
        trace,
        n,
        maxiters,
        iterations,
        trace_io = stdout;
        trace_level = SnoptTraceMinimal(),
        store_trace = Val(false),
        lb = fill(-Inf, n),
        ub = fill(Inf, n),
        algorithm = nothing,
        sense = OptimizationBase.MinSense,
        ws_rw = Float64[]
    )
    return SnoptProgressLogger(
        progress,
        callback,
        trace,
        snopt_store_trace(store_trace),
        trace_io,
        normalize_trace_level(trace_level),
        algorithm,
        sense,
        n,
        maxiters,
        iterations,
        Ref(0),
        Ref(false),
        Ref(false),
        zeros(n),
        Ref(false),
        Float64.(lb),
        Float64.(ub),
        Ref(NaN),
        Ref(NaN),
        ws_rw isa Vector{Float64} ? ws_rw : Float64.(ws_rw),
        Ref(0),
        SnoptTraceEntry[]
    )
end

# The logger is needed only for one of these; without it SNOPT runs without `snstop`.
function uses_snstop(cb::SnoptProgressLogger)
    return cb.show_trace || cb.store_trace || cb.progress || cb.callback !== nothing
end

function trace_algorithm_block(algorithm)
    io = IOBuffer()
    println(io, "Algorithm: SnoptOptimizer(")
    @printf io "    %-30s = %s\n" "hessian" repr(algorithm.hessian)
    @printf io "    %-30s = %d\n" "major_iterations_limit" algorithm.major_iterations_limit
    @printf io "    %-30s = %.8e\n" "major_optimality_tolerance" algorithm.major_optimality_tolerance
    @printf io "    %-30s = %.8e\n" "major_feasibility_tolerance" algorithm.major_feasibility_tolerance
    println(io, ")")
    return String(take!(io))
end

function print_trace_algorithm(io::IO, algorithm)
    algorithm === nothing && return nothing
    # Base.have_color is `nothing` until Julia resolves the terminal's color
    # support, which never happens for redirected/non-TTY output (CI, Pkg.test).
    # Coalesce to `false` so the IOContext color flag is always a Bool; passing
    # `nothing` makes printstyled throw "non-boolean (Nothing) used in boolean
    # context".
    color = get(io, :color, something(Base.have_color, false))
    printstyled(IOContext(io, :color => color), trace_algorithm_block(algorithm); color = :green)
    println(io)
    return nothing
end

trace_integer(value::Int) = value < 0 ? "-" : string(value)
trace_float(value::Real) = isfinite(value) ? @sprintf("%.8e", Float64(value)) : "-"

function print_trace_header(io::IO, level::SnoptTraceLevel)
    if level.trace_mode === :all
        @printf io "%5s %5s %14s %14s %14s %14s %14s\n" "-----" "-----" "--------------" "--------------" "--------------" "--------------" "--------------"
        @printf io "%5s %5s %14s %14s %14s %14s %14s\n" "Major" "Minor" "Objective" "Constr viol" "Optimality" "Step" "Bound viol"
        @printf io "%5s %5s %14s %14s %14s %14s %14s\n" "-----" "-----" "--------------" "--------------" "--------------" "--------------" "--------------"
    else
        @printf io "%5s %5s %14s %14s %14s %14s\n" "-----" "-----" "--------------" "--------------" "--------------" "--------------"
        @printf io "%5s %5s %14s %14s %14s %14s\n" "Major" "Minor" "Objective" "Constr viol" "Optimality" "Step"
        @printf io "%5s %5s %14s %14s %14s %14s\n" "-----" "-----" "--------------" "--------------" "--------------" "--------------"
    end
    return nothing
end

function print_trace_header(cb::SnoptProgressLogger)
    cb.trace_header_printed[] && return nothing
    print_trace_algorithm(cb.trace_io, cb.algorithm)
    print_trace_header(cb.trace_io, cb.trace_level)
    cb.trace_header_printed[] = true
    return nothing
end

function print_trace_entry(io::IO, entry::SnoptTraceEntry, level::SnoptTraceLevel)
    major = trace_integer(entry.major_iter)
    minor = trace_integer(entry.minor_iter)
    objective = trace_float(entry.objective)
    constraint_violation = trace_float(entry.constraint_violation)
    optimality = trace_float(entry.optimality)
    step_norm = trace_float(entry.step_norm)
    bound_violation = trace_float(entry.bound_violation)

    if entry.iteration < 0
        if level.trace_mode === :all
            @printf io "%5s %5s %14s %14s %14s %14s %14s\n" major minor objective constraint_violation optimality "-" bound_violation
        else
            @printf io "%5s %5s %14s %14s %14s %14s\n" major minor objective constraint_violation optimality "-"
        end
    elseif level.trace_mode === :all
        @printf io "%5s %5s %14s %14s %14s %14s %14s\n" major minor objective constraint_violation optimality step_norm bound_violation
    else
        @printf io "%5s %5s %14s %14s %14s %14s\n" major minor objective constraint_violation optimality step_norm
    end
    return nothing
end

function snopt_workspace_value(values::Vector{Float64}, index::Int)
    return length(values) >= index ? values[index] : NaN
end

function current_optimality(cb::SnoptProgressLogger)
    isfinite(cb.last_optimality[]) && return max(cb.last_optimality[], 0.0)
    value = snopt_workspace_value(cb.ws_rw, SNOPT_DUAL_INFEASIBILITY_INDEX)
    return isfinite(value) ? max(value, 0.0) : NaN
end

function finite_violation(value::Real, lower::Real, upper::Real)
    lower_violation = isfinite(lower) ? max(Float64(lower) - Float64(value), 0.0) : 0.0
    upper_violation = isfinite(upper) ? max(Float64(value) - Float64(upper), 0.0) : 0.0
    return max(lower_violation, upper_violation)
end

function bounds_violation(x::AbstractVector, lb::AbstractVector, ub::AbstractVector)
    violation = 0.0
    for i in eachindex(x)
        violation = max(violation, finite_violation(x[i], lb[i], ub[i]))
    end
    return violation
end

function trace_show_now(cb::SnoptProgressLogger, entry::SnoptTraceEntry)
    cb.show_trace || return false
    entry.iteration < 0 && return true
    return mod1(entry.iteration, cb.trace_level.print_frequency) == 1
end

function trace_store_now(cb::SnoptProgressLogger, entry::SnoptTraceEntry)
    cb.store_trace || return false
    entry.iteration < 0 && return true
    return mod1(entry.iteration, cb.trace_level.store_frequency) == 1
end

function emit_trace_entry!(cb::SnoptProgressLogger, entry::SnoptTraceEntry)
    if trace_show_now(cb, entry)
        print_trace_header(cb)   # algorithm block + column header on first call
        rows = cb.rows_printed[]
        if entry.iteration >= 0 && rows > 0 && mod(rows, TRACE_HEADER_REPEAT_INTERVAL) == 0
            print_trace_header(cb.trace_io, cb.trace_level)
        end
        print_trace_entry(cb.trace_io, entry, cb.trace_level)
        entry.iteration >= 0 && (cb.rows_printed[] += 1)
        flush(cb.trace_io)
    end
    trace_store_now(cb, entry) && push!(cb.trace_history, entry)
    return nothing
end

function trace_major_iteration!(cb::SnoptProgressLogger, event::SNOPT.SnoptStopEvent, objective)
    (cb.show_trace || cb.store_trace) || return nothing
    cb.trace_evals[] += 1
    # Only SnoptTraceAll keeps copies of the point and the constraint values.
    keep_vectors = cb.trace_level.trace_mode === :all
    nc = min(event.nncon, length(event.fcon))
    entry = SnoptTraceEntry(
        cb.trace_evals[],
        event.major_iter,
        event.minor_iter,
        objective,
        event.step,
        event.primal_infeasibility,
        event.dual_infeasibility,
        bounds_violation(cb.u, cb.lb, cb.ub),
        keep_vectors ? copy(cb.u) : nothing,
        keep_vectors ? event.fcon[1:nc] : nothing,
        nothing
    )
    emit_trace_entry!(cb, entry)
    return nothing
end

function (cb::SnoptProgressLogger)(event::SNOPT.SnoptStopEvent)
    copyto!(cb.u, view(event.x, 1:cb.n))
    cb.has_u[] = true
    cb.iterations[] = event.major_iter
    cb.last_constraint_violation[] = event.primal_infeasibility
    cb.last_optimality[] = event.dual_infeasibility
    objective = cb.sense === OptimizationBase.MaxSense ? -event.objective : event.objective

    trace_major_iteration!(cb, event, objective)

    if cb.progress
        # Without `maxiters`, the fraction is taken of the optimizer's major iteration limit.
        limit = if cb.maxiters !== nothing
            cb.maxiters
        elseif cb.algorithm !== nothing
            cb.algorithm.major_iterations_limit
        else
            nothing
        end
        limit === nothing ||
            Base.@logmsg(Base.LogLevel(-1), "", progress = event.major_iter / limit)
    end

    if cb.callback !== nothing
        opt_state = OptimizationBase.OptimizationState(;
            iter = event.major_iter, u = cb.u, objective, original = event
        )
        return !cb.callback(opt_state, objective)  # `true` from the user stops SNOPT
    end
    return true
end

function finish_trace!(
        cb::SnoptProgressLogger,
        status::Int,
        objective::Float64;
        major_iter::Int = cb.iterations[],
        minor_iter::Int = 0
    )
    (cb.show_trace || cb.store_trace) || return nothing
    cb.final_trace_printed[] && return nothing
    final_entry = SnoptTraceEntry(
        -1,
        major_iter,
        minor_iter,
        objective,
        NaN,
        cb.last_constraint_violation[],
        current_optimality(cb),
        cb.has_u[] ? bounds_violation(cb.u, cb.lb, cb.ub) : NaN,
        nothing,
        nothing,
        SNOPT.status_from_code(status)
    )
    emit_trace_entry!(cb, final_entry)
    cb.final_trace_printed[] = true
    return nothing
end

function stored_trace(cb::SnoptProgressLogger)
    cb.store_trace || return nothing
    return SnoptTrace(copy(cb.trace_history), cb.trace_level)
end
