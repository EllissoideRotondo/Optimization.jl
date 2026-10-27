# OptimizationSNOPT.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/Optimization/stable/optimization_packages/snopt/)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

OptimizationSNOPT.jl is a component package of [Optimization.jl](https://github.com/SciML/Optimization.jl) in the SciML ecosystem. It holds the wrapper for [SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/), a commercial sequential quadratic programming solver for large, sparse, constrained nonlinear problems, and calls it through [SNOPT.jl](https://github.com/EllissoideRotondo/SNOPT.jl).

OptimizationSNOPT.jl needs Julia 1.10 or later and a licensed SNOPT 7.7 library, `libsnopt7`. Neither package includes the library. Set `SNOPTDIR` to the directory that contains it before starting Julia.

```julia
using Optimization, OptimizationSNOPT, ForwardDiff

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
f = OptimizationFunction(rosenbrock, AutoForwardDiff())
prob = OptimizationProblem(f, [0.0, 0.0], [1.0, 100.0])
sol = solve(prob, SnoptOptimizer())
```
