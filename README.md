# solver

The optimisation core extracted from [pleaser](https://github.com/x21-dev/pleaser):
deterministic CMA-ES stages and a quality-diversity archive over a problem
interface. So far: the interface (`source/solver/problem.d`), CMA-ES
(`cmaes.d`) and its RNG (`rng.d`), moved from pleaser unchanged, and the
CMA-ES stage over any problem (`stage.d`: stall stop, optional surrogate
pre-screening through a `StageModel`, a per-generation callback, and stages
that stop and resume bit for bit, their state as plain data in `StageRun`). Also the
surrogate ensemble (`ensemble.d`: small ReLU networks trained by Adam,
deterministic on any thread count) and `bytes.d`, the little-endian
encoding of saved states. And the archive (`archive.d`: CMA-MAE over a
problem's descriptors on a grid, with explorers that head for cells not
discovered yet, model-guided and
pre-screened emitters, focus, and runs that stop and resume bit for bit,
their state between iterations as plain data in `RunState`; its ranking
and its restart and explorer rules, `ArchiveRules`, are options). Also the learned
model (`model.d`: the ensemble trained on a run's evaluations, for the
pre-screened stages and the model-guided emitters, over a problem's
surrogate members). And `detmath.d`: `log2` and `exp2` (scalar and 4-lane) in plain IEEE
arithmetic, the same bits on every platform, copied from pleaser's
`pleaser.detmath`. A problem's seeded noise should come from `rng.d`'s
`SplitMix64`.

Runs are bit-identical on any thread count of one platform, not across
platforms (the C maths library differs), so keep golden runs per
platform; see the manual's Determinism section.

solver is a source library: a dependent compiles its modules with its own,
so calls into it inline as calls within one package do.

```
dub test --compiler=ldc2
```

To use solver in another project, see the [developer manual](docs/manual.md)
and its worked example in `docs/example`.
