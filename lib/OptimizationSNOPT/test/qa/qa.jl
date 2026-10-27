using SciMLTesting, OptimizationSNOPT, JET
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * the ignored *_are_public names are owned by SciMLBase, OptimizationBase, SNOPT or
#    Base and are not (yet) declared public; the proper fix is upstream `public`
#    declarations, not a local change.
run_qa(
    OptimizationSNOPT;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_are_public = (; ignore = (Symbol("@logmsg"), :DEFAULT_VERBOSE, :LogLevel, :OptimizationState, :ReInitCache, :SNOPT_LOCK, :T, :get_observed, :get_p, :get_paramsyms, :get_syms, :have_color, :instantiate_function, :supports_sense)),
    ),
)
