mutable struct SnoptCache{
        T, F <: OptimizationFunction, RC, LB, UB, I, S,
        JT <: AbstractMatrix{T}, CB, O,
    } <: SciMLBase.AbstractOptimizationCache
    const f::F
    const n::Int
    const num_cons::Int
    const reinit_cache::RC
    const lb::LB
    const ub::UB
    const int::I
    const lcons::Vector{T}
    const ucons::Vector{T}
    const sense::S
    J::JT
    const callback::CB
    const progress::Bool
    f_calls::Int
    f_grad_calls::Int
    const iterations::Ref{Int}
    obj_expr::Union{Expr, Nothing}
    cons_expr::Union{Vector{Expr}, Nothing}
    const opt::O
    const solver_args::NamedTuple
end

function Base.getproperty(cache::SnoptCache, name::Symbol)
    if name in fieldnames(OptimizationBase.ReInitCache)
        return getfield(cache.reinit_cache, name)
    end
    return getfield(cache, name)
end
function Base.setproperty!(cache::SnoptCache, name::Symbol, x)
    if name in fieldnames(OptimizationBase.ReInitCache)
        return setfield!(cache.reinit_cache, name, x)
    end
    return setfield!(cache, name, x)
end

function SciMLBase.get_p(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {T, N, uType, C <: SnoptCache}
    return sol.cache.p
end
function SciMLBase.get_observed(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {T, N, uType, C <: SnoptCache}
    return sol.cache.f.observed
end
function SciMLBase.get_syms(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {T, N, uType, C <: SnoptCache}
    return variable_symbols(sol.cache.f)
end
function SciMLBase.get_paramsyms(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {T, N, uType, C <: SnoptCache}
    return parameter_symbols(sol.cache.f)
end

function SnoptCache(
        prob, opt;
        callback = nothing,
        progress = false,
        verbose = OptimizationBase.DEFAULT_VERBOSE,
        trace_level = SnoptTraceMinimal(),
        store_trace = Val(false),
        kwargs...
    )
    if prob.int !== nothing && any(prob.int)
        # Follow OptimizationIpopt's convention: solve the continuous
        # relaxation, but say so instead of silently dropping integrality.
        @warn "SNOPT is a continuous NLP solver; integer variables (`int` in " *
            "OptimizationProblem) are relaxed and solved as continuous"
    end
    reinit_cache = OptimizationBase.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`
    show_trace = snopt_show_trace(verbose)
    final_verbose = verbose

    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = OptimizationBase.instantiate_function(
        prob.f, reinit_cache, prob.f.adtype, num_cons;
        g = true, cons_j = true
    )
    T = eltype(prob.u0)
    n = length(prob.u0)

    J = if isnothing(f.cons_jac_prototype)
        zeros(T, num_cons, n)
    else
        similar(f.cons_jac_prototype, T)
    end
    lcons = prob.lcons === nothing ? fill(T(-Inf), num_cons) : prob.lcons
    ucons = prob.ucons === nothing ? fill(T(Inf), num_cons) : prob.ucons

    sys = f.sys isa SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing} ?
        nothing : f.sys
    obj_expr = f.expr
    cons_expr = f.cons_expr

    solver_args = merge(
        NamedTuple(kwargs),
        (;
            verbose = final_verbose, show_trace,
            trace_level = normalize_trace_level(trace_level), store_trace,
        )
    )

    return SnoptCache(
        f,
        n,
        num_cons,
        reinit_cache,
        prob.lb,
        prob.ub,
        prob.int,
        lcons,
        ucons,
        prob.sense,
        J,
        callback,
        progress,
        0,
        0,
        Ref(0),
        obj_expr,
        cons_expr,
        opt,
        solver_args
    )
end

# Pass the current `cache.p`, which `reinit!` may replace, to the instantiated
# derivative functions; without it they keep the parameters from `init`. As in
# OptimizationIpopt, the `AutoSymbolics` wrappers hold the parameters themselves, and
# `instantiate_function` turns `nothing` into `NullParameters()`, so those cases are
# called without `p`.
function accepts_live_p(cache::SnoptCache)
    adtype = cache.f.adtype
    return !(
        adtype isa ADTypes.AutoSymbolics ||
            (adtype isa ADTypes.AutoSparse && adtype.dense_ad isa ADTypes.AutoSymbolics) ||
            cache.p === nothing ||
            cache.p isa SciMLBase.NullParameters
    )
end

function eval_objective(cache::SnoptCache, x)
    l = cache.f(x, cache.p)
    cache.f_calls += 1
    return cache.sense === OptimizationBase.MaxSense ? -l : l
end

function eval_constraint(cache::SnoptCache, g, x)
    if accepts_live_p(cache)
        cache.f.cons(g, x, cache.p)
    else
        cache.f.cons(g, x)
    end
    return
end

function eval_objective_gradient(cache::SnoptCache, G, x)
    if cache.f.grad === nothing
        error(
            "Use OptimizationFunction to pass the objective gradient or " *
                "automatically generate it with one of the autodiff backends. " *
                "If you are using the ModelingToolkit symbolic interface, pass the `grad` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    if accepts_live_p(cache)
        cache.f.grad(G, x, cache.p)
    else
        cache.f.grad(G, x)
    end
    cache.f_grad_calls += 1

    if cache.sense === OptimizationBase.MaxSense
        G .*= -one(eltype(G))
    end

    return
end

function eval_constraint_jacobian(cache::SnoptCache, j, x)
    if isempty(j)
        return
    elseif cache.f.cons_j === nothing
        error(
            "Use OptimizationFunction to pass the constraints' jacobian or " *
                "automatically generate it with one of the autodiff backends. " *
                "If you are using the ModelingToolkit symbolic interface, pass the `cons_j` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    # Get and cache the Jacobian object here once. `evaluator.J` calls
    # `getproperty`, which is expensive because it calls `fieldnames`.
    J = cache.J
    cons_j = cache.f.cons_j
    p = cache.p
    # Sparse instantiation can capture `p` in a two-argument `cons_j` wrapper.
    if accepts_live_p(cache) && applicable(cons_j, J, x, p)
        cons_j(J, x, p)
    else
        cons_j(J, x)
    end
    if J isa SparseMatrixCSC
        nnz = nonzeros(J)
        length(j) == length(nnz) ||
            throw(
            DimensionMismatch(
                "SNOPT requested $(length(j)) Jacobian nonzeros, but the cached sparse Jacobian has $(length(nnz))"
            )
        )
        for (i, Ji) in zip(eachindex(j), nnz)
            j[i] = Ji
        end
    else
        j .= vec(J)
    end
    return
end
