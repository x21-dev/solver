# solver

The optimisation core extracted from [pleaser](https://github.com/x21-dev/pleaser):
deterministic CMA-ES stages and a quality-diversity archive over a problem
interface. So far: the interface (`source/solver/problem.d`), CMA-ES
(`cmaes.d`) and its RNG (`rng.d`), moved from pleaser unchanged, and the
CMA-ES stage over any problem (`stage.d`: stall stop, optional surrogate
pre-screening through a `StageModel`, a per-generation callback).

```
dub test --compiler=ldc2
```
