# solver: developer manual

solver is a deterministic black-box optimiser for D, built with LDC. You
describe a problem: genes in [0, 1], an evaluation type of your own,
an objective to maximise, and limits in three tiers. solver searches it
with two tools:

- **A CMA-ES stage** (`solver.stage`) finds the best candidate from a
  start point. It stops early when progress stalls, and a learned model
  can pre-screen its candidates.
- **An archive** (`solver.archive`) runs CMA-MAE. It keeps the best
  candidate in every cell of a grid over your problem's descriptors. It
  has explorers that head for empty cells, and optional model-guided and
  pre-screened emitters. A run can stop between iterations and resume bit
  for bit.

Both tools are deterministic. The same problem, seed and options give the
same bits on any thread count of one platform, provided the problem keeps
the rules in [Determinism](#determinism). Across platforms runs differ.

This manual describes solver main as of 2026-10-29: the extraction from
pleaser finished (solver#1 to #8), CI on Linux and Windows (#10), the
checkpoint template (#11) and `solver.detmath` (#12). Everything here was
checked against that code. The worked example in
[`docs/example`](example/source/app.d) builds and runs against it, and CI
runs it on Linux and Windows.
[Gaps and caveats](#gaps-and-caveats) lists what solver doesn't do yet and
the traps in the current API. Read it before you rely on a feature.

Contents:

1. [Adding solver to a project](#adding-solver-to-a-project)
2. [The model: genes, evaluations, limits, ranking](#the-model)
3. [Writing a problem](#writing-a-problem)
4. [A CMA-ES stage](#a-cma-es-stage)
5. [The archive](#the-archive)
6. [A learned model](#a-learned-model)
7. [Stopping, saving and resuming](#stopping-saving-and-resuming)
8. [Lower-level pieces](#lower-level-pieces)
9. [Determinism](#determinism)
10. [Performance](#performance)
11. [Gaps and caveats](#gaps-and-caveats)
12. [Module reference](#module-reference)

---

## Adding solver to a project

solver is a dub **source library** (`"targetType": "sourceLibrary"`). Your
project compiles solver's modules together with its own, so calls into
solver can inline like calls within one package. Add it as a dependency:

```json
"dependencies": {
    "solver": { "path": "../solver" }
}
```

Use a path dependency (pleaser checks solver out at `../solver`) or a git
dependency with a pinned commit. solver isn't on the dub registry.

Requirements:

- **LDC only.** `solver.ensemble` imports `ldc.simd`, so DMD and GDC can't
  build any module that imports it: `solver.ensemble`, `solver.model`, or
  anything that imports those. solver is developed with LDC 1.42.0.
- No other dependencies. solver uses Phobos and `ldc.simd` only.
- solver has no `package.d`. Import the modules you need by name:
  `import solver.problem; import solver.stage; import solver.archive;`
  and so on.

Build flags are your project's. solver adds none. pleaser builds with
`--build=release-native`. If you change the target CPU, check that the
floating-point results stay identical across the machines you need
(see [Determinism](#determinism)).

Run solver's own tests with:

```
dub test --compiler=ldc2
```

Build and run the worked example with:

```
cd docs/example && dub run --compiler=ldc2 --build=release
```

---

## The model

### Gene space

Every search works on `dimension` genes. Each gene is meant to lie in
[0, 1]. Your problem maps genes to physical values (millimetres, Hz, a
count) itself. `GeneSpec` helps with that (see [Genes](#genes-genespec)).

CMA-ES samples a normal distribution, so its candidates can fall outside
[0, 1]. **`evaluate` gets genes exactly as CMA-ES sampled them, which can
be outside [0, 1].** Your problem must clamp them (`clamp01`, or
`GeneSpec.toPhysical`, which clamps). The search ranks a candidate with
genes out of range lower, by `Ranking.boundPenalty` times `boundExcess(x)`
(the sum of the squared overshoots). Everything after the evaluation gets
the genes clamped to [0, 1]: the best, the archive's elites, the model,
and the progress reports.

### Evaluations

You choose the evaluation type, `P.Eval`. solver never looks inside it.
It reads it only through your problem's members: `scored`, `objective`,
`breach` and `descriptors`. `P.Eval` must be a **value type**: a copy must
not alias anything your problem changes later. The archive copies an elite
with `copyOf`, which duplicates arrays one level deep. So a struct of
scalars and arrays of scalars is safe, but an array of arrays, a class
reference or a pointer is shared between copies.

### Limits: three tiers

`Breach` holds how far an evaluation is over its limits in each tier. Each
field is 0 within the limits and never negative.

| Tier | Meaning | How it ranks |
|---|---|---|
| `hard` | Structure and survival | Any hard breach ranks below every candidate without one |
| `terminal` | Conditions on the outcome | Ranks after hard, before the objective |
| `soft` | Limits that may give, at a cost | Subtracted from the objective: `value = objective − soft` |

A candidate is **feasible** when `hard == 0 && terminal == 0`. Only
feasible candidates enter the archive or become a stage's best. Soft limits
never make a candidate infeasible. They only lower its value.

Your problem computes `Breach` itself, adding each tier up in the order it
chooses, so it controls the bits. If it has no order of its own to keep,
`breachOf(specs, violations)` adds per-limit violations in spec order, with
soft ones weighted by `ConstraintSpec.weight`.

### Ranking

CMA-ES minimises a fitness. `Ranking.fitness(breach, value, excess)` gives:

```
hard > 0      →  hardOffset + hard
terminal > 0  →  terminalOffset + terminal
otherwise     →  −value
              +  boundPenalty × excess
```

The defaults are `hardOffset = 100`, `terminalOffset = 50` and
`boundPenalty = 10`. They are pleaser's settings, for values from 0 to 28.
**The offsets must put every breach above every value.**
`terminalOffset` must exceed minus the lowest value you can produce, and
`hardOffset` must exceed `terminalOffset` plus the largest terminal breach.
If your objective can be below −50 or above +50, set your own `Ranking`
(`StageOptions.ranking`, or a `ranking` field in your archive options type;
see [Options](#options)).

The archive ranks an emitter's candidates the same way, but uses the
CMA-MAE improvement (value minus the cell's threshold) or the explorer's
distance in place of the value.

### Scoring

`evaluate` takes a `Scoring`. With `Scoring.feasibleOnly`, a candidate
over a hard limit may skip computing its objective: the ranking never reads
it. Report that through `scored(e) == false`. Both searches always call
`evaluate` with `feasibleOnly`. `Scoring.all` is there for your own calls,
such as a model's training data.

---

## Writing a problem

A problem is a compile-time concept, not a class. The searches are
templates over `P`, so every call into your problem is direct and can
inline. `isProblem!P` checks the required members.

### Required members

```d
struct MyProblem
{
    struct Eval { ... }            // or alias Eval = SomeType;

    size_t dimension() const;      // the number of genes
    Eval[] evaluate(const(double[])[] genes, Scoring scoring);
    bool   scored(const ref Eval e) const;      // objective computed?
    double objective(const ref Eval e) const;   // higher is better; read only when scored
    Breach breach(const ref Eval e) const;      // per tier, 0 within
}
static assert(isProblem!MyProblem);
```

`evaluate` gets a whole batch (a stage's population, or every emitter's
population in an archive iteration) and returns results **in input
order**. It may run the batch in parallel. It must be deterministic: the
same genes must give the same bits whatever the batch size, whatever the
other candidates in the batch, and on any thread count. It is the only
non-`const` member the searches call. The problem is passed by `ref`, so
it may count calls or cache.

### Optional members

The searches detect each one with `__traits(hasMember)` or a concept
template. This table lists only the members that the code reads.

| Member | Read by | Purpose |
|---|---|---|
| `size_t descriptorCount() const`, `void descriptors(const ref Eval, double[] d) const` | Archive (required there: `hasDescriptors!P`) | Where an evaluation lands. The grid maps these values to a cell. Read only for feasible candidates. |
| `alias Causes`, `Causes newCauses() const`, `void countCauses(ref Causes, const ref Eval) const` | Stage and archive (`hasCauses!P`, in `solver.stage`) | Counts why candidates broke their limits, for progress reports. `Causes` needs `opOpAssign!"+"`. |
| `double screenPenalty(const(double)[] genes)` | Pre-screening in the stage and the archive | A known cost subtracted from a predicted objective (pleaser's processing cost, which needs no render). |
| `string noFeasibleSeed() const` | Archive | A hint appended to `NoFeasibleSeed`'s message. |
| `void adoptCauses(ref Causes)` | Archive, on resume | Lets the problem fix up resumed cause counts. |

`solver.problem` also documents the surrogate members `featureCount`,
`features`, `targetCount`, `targets` and `predicted` (`hasSurrogate!P`):
with them a problem is its own learning spec for the learned model (see
[A learned model](#a-learned-model)). And it documents `fingerprint` and
`saveEval`/`loadEval` (`isSaveable!P`), a convention for a project's own
save format: no search reads them, and saving is up to you (see
[To a file](#to-a-file)).

### Genes: `GeneSpec`

```d
static immutable GeneSpec[2] spec = [GeneSpec("width", 1, 10, 5),
                                     GeneSpec("rate", 0.1, 100, 1, Scale.log)];
immutable double w = spec[0].toPhysical(g[0]);   // clamps, then maps
immutable double x0 = spec[0].toGene(spec[0].neutral);
```

`GeneSpec(name, lo, hi, neutral, scale)`:

- `Scale.linear` maps 0 to `lo` and 1 to `hi`.
- `Scale.log` needs `lo > 0`.
- `Scale.integer` rounds to the nearest whole value in [lo, hi].
- `fixed` is true when `lo == hi`. Leave a fixed gene out of the search.
  `toGene` asserts on one.

`toPhysical` clamps its gene to [0, 1], so a problem that maps every gene
through it is safe from out-of-range genes.

### Limits: `ConstraintSpec` and `breachOf`

```d
static immutable ConstraintSpec[3] limits = [ConstraintSpec("load", Tier.hard),
                                             ConstraintSpec("min_width", Tier.terminal),
                                             ConstraintSpec("rate_cap", Tier.soft, 0.05)];
immutable double[3] v = [w * r > 200 ? w * r - 200 : 0, w < 2 ? 2 - w : 0, r > 40 ? r - 40 : 0];
e.breach = breachOf(limits[], v[]);
```

`ConstraintSpec.group` is a free label for your own cause counts. solver
doesn't read it.

### The worked example's problem

[`docs/example/source/app.d`](example/source/app.d) has a two-gene problem,
`Beam`. It has one limit in each tier, cause counts, descriptors, a
fingerprint and a `noFeasibleSeed` hint. Start from a copy of it.

### Dynamic problems

`DynamicProblem` is an abstract class for problems behind virtual calls,
such as a test, a problem chosen at run time or a binding to another
language of your own (solver has no bindings). It satisfies
`isProblem`, `hasDescriptors` and `isSaveable`. Its `Eval` is `Outcome`,
which has `scored`, `objective`, `violations`, `breach`, `descriptors` and
`targets`. Override `genes`, `constraints`, `fingerprint`, `evaluateInto`,
and `descriptorCount` if you use the archive. Each call covers a whole
batch, so the dispatch cost is small next to an evaluation.

---

## A CMA-ES stage

```d
import solver.stage;

Beam p;
Best!(Beam.Eval) best;                          // carried across stages
double[] x0 = [...];                            // start genes
StageOptions so = {lambda: 8, sigma: 0.2, budget: 800, seed: 42};
const end = cmaStage(p, best, x0, so, null, (const ref StageGeneration!Beam g) {
    // after each generation
});
// best.eval, best.genes (clamped), best.feasible, best.evals
```

`cmaStage(ref P p, ref Best!(P.Eval) best, const(double)[] x0, const ref
StageOptions o, M screenModel, generationDone = null)` runs CMA-ES from
`x0` for `o.budget` evaluations, rounded up to whole generations.
`screenModel` is `null` or a `StageModel!(P.Eval)`.

**`StageOptions`:**

| Field | Default | Meaning |
|---|---|---|
| `lambda` | none | Population size. Must be > 0 (`cmaStage` throws otherwise). |
| `sigma` | none | Initial step size, in gene units |
| `budget` | none | Evaluations, rounded up to whole generations |
| `seed` | none | CMA-ES and pre-screening seed |
| `stallGenerations`, `stallTolerance` | 100, 0.01 | Stop when the best value rose by less than the tolerance over the last N generations (0 = never) |
| `surrogatePool` | 0 | With a model: sample this many times `lambda`, evaluate `lambda`. Must be ≥ 1 when you pass a model (`cmaStage` throws otherwise). |
| `surrogateShare` | 0.5 | Share of the evaluated candidates that the model picks (the rest are random) |
| `surrogateFrom` | 5 | Generation where pre-screening starts |
| `surrogateEpochs` | 10 | Epochs trained before each pre-screened generation |
| `ranking` | `Ranking.init` | See [Ranking](#ranking) |

**`Best!E`** is the best feasible candidate so far. Start a run with
`Best.init`, or set `genes` and `eval` to your own start. Pass the same
`Best` to later stages: each stage updates it only with a better feasible
candidate. Its fields:

- `feasible` stays false until some candidate is feasible.
- `evals` counts the evaluations of every stage.

To chain stages, pass `best.genes` as the next stage's `x0`, perhaps with a
smaller `sigma`.

**`StageGeneration!P`** is passed to the callback after each generation,
with these fields:

- `generation` counts from 1.
- `before` is the number of evaluations before this generation.
- `genes` and `evals` are this generation's candidates, genes clamped.
- `best` is the best value so far, −∞ until a candidate is feasible.
- `populationBest` is this generation's best value, NaN if none is
  feasible.
- `infeasible`, `sigma`.
- `causes`, when `hasCauses!P`.
- The pre-screening fields `picked`, `byModel`, `sampled` and
  `predictedFeasible`. `picked` is null when a generation isn't
  pre-screened.

**`StageEnd`** returns `generations`, `spent`, `stalled`, and `rise` (the
best's rise over the window it stalled in).

**Pre-screening.** From generation `surrogateFrom` on, the stage trains the
model, samples `surrogatePool × lambda` candidates and ranks them on the
model's prediction (minus `screenPenalty`). It evaluates `lambda` of them:
`surrogateShare` picked by the model and the rest at random. CMA-ES is then
told the real fitness of the evaluated candidates and the predicted fitness
of the others. A `StageModel!E` has three members:

```d
interface StageModel(E)
{
    void learn(const(double[])[] genes, const(E)[] evals);
    void train(size_t epochs);
    void predictScore(const(double[])[] genes, double[] score, double[] violation);
}
```

`violation` is ranked as a hard breach. `LearnedModel` implements this
interface (see [A learned model](#a-learned-model)).

---

## The archive

### Grids

A grid maps descriptor values to cells. `isGrid!G` requires these members:

| Member | Purpose |
|---|---|
| `size_t cells() const` | The number of cells |
| `size_t cellOf(const(double)[] d) const` | The cell of descriptors `d` |
| `double distance(const(double)[] d, size_t cell) const` | How far `d` is from `cell`: 0 inside, in bin widths. Explorers move down it. |
| `long binDistance2(size_t a, size_t b) const` | Squared distance between cells in bins. Used for explorer starts and `selectDiverse`. |
| `string code(size_t) const`, `size_t parse(string) const` | A cell's name in files and on a command line |

`BinnedGrid` is the built-in grid. It's a product of `Axis` values, with
the first axis varying fastest in the cell index:

```d
static immutable Axis[2] axes = [Axis("width", [3, 5, 7], "nmwW", 0, 2),
                                 Axis("rate", [1, 10, 40], "slfF", 0, 10)];
const grid = BinnedGrid(axes[]);      // 4 × 4 = 16 cells, codes like "wf"
```

`Axis(name, edges, letters, nanBin, width)` has these parts:

- `edges` are ascending. A value below `edges[0]` goes in bin 0, and a
  value from the last edge up goes in the last bin.
- `letters` has one character per bin (`edges.length + 1`).
- `nanBin` is the bin a NaN value goes in.
- `width` is the unit of `distance` on this axis. Set it to a typical bin
  width so that the axes weigh alike.

If your cells aren't a regular product of bins, bring your own `G`.
pleaser's character grid is an example.

### Running

```d
import solver.archive;

alias A = Archiving!(Beam, BinnedGrid);         // P, G, O = ArchiveOptions, N = 0, minScore = 0
ArchiveOptions ao = {evals: 4000, emitters: 4, explorers: 2, lambda: 8, seed: 9};
A.Run run = A.runArchive(p, grid, seeds, ao);
```

`Archiving!(P, G, O, N, minScore)` is a template of functions.
`P` must satisfy `hasDescriptors!P` and `G` must satisfy `isGrid!G`. It
has these members:

- `Run` is `ArchiveRun!(P, O, N, minScore)`.
- `Generation` is `ArchiveGeneration!(CausesOf!P)`.
- `Batch` is `ArchiveBatch!(P.Eval)`.
- `Model` is `ArchiveSurrogate!(P.Eval)`.
- `runArchive`:

```d
Run runArchive(ref P p, const ref G grid, const(double[])[] seeds, O opts,
               scope void delegate(const ref Generation) progress = null,
               Run* resume = null,
               scope bool delegate(const ref Run) afterIteration = null,
               scope void delegate(const ref Batch) evaluated = null,
               Model model = null);
```

A run goes like this:

1. The seeds are evaluated, and the feasible ones enter the archive. If
   none is feasible, the run throws `NoFeasibleSeed` (with your
   `noFeasibleSeed()` hint).
2. All emitters start from the best seed. The last `explorers` of them are
   explorers.
3. Each iteration, every emitter asks for `lambda` candidates. All
   `emitters × lambda` are evaluated in **one** `evaluate` call. Then the
   candidates are offered to the archive emitter by emitter, in population
   order.
4. Improvers rank candidates by CMA-MAE improvement: value minus the cell's
   threshold. An improver restarts from a random elite when it has been
   idle for `RESTART_GENERATIONS` generations, when nearly all its
   candidates are infeasible, or when its step size collapses.
5. An explorer heads for the empty cell nearest the occupied ones, down
   `grid.distance`. It retargets when it reaches the cell or gets stuck.
   A cell it gives up on is blocked for `EXPLORE_BLOCK` iterations.
6. The run stops once `opts.evals` evaluations beyond the seeds are spent
   (in whole iterations), or when `afterIteration` returns true.

**Template parameters `N` and `minScore`.**

- `N = 0` (the default) holds the cells in arrays sized by `grid.cells`.
- `N > 0` fixes the cell count at compile time, and the archive holds
  `Elite!E[N]` by value. That makes `Run` large; read
  [Performance](#performance) before you use it.
- `minScore` is where every cell's threshold starts, and the zero of
  `qdScore`. Set it at or below the lowest feasible value.

### Options

`ArchiveOptions` fields (defaults in brackets):

| Field | Meaning |
|---|---|
| `evals` [8000] | Budget beyond the seeds, rounded up to whole iterations. On resume it is still the whole run's budget. |
| `emitters` [5] | Emitters, explorers included |
| `lambda` [0] | Population per emitter. 0 uses CMA-ES's default for the dimension (4 + ⌊3 ln n⌋). |
| `sigma` [0.05] | Improvers' initial step size |
| `alpha` [0.1] | Archive learning rate. 0 optimises value alone; 1 is CMA-ME. |
| `seed` [1] | Run seed |
| `explorers` [2] | How many of the emitters are explorers (the last ones) |
| `explorerSigma` [0.1] | Explorers' step size |
| `surrogatePool` [0] | With a model, model-guided explorers sample this many times their population. 0 = off. **A model must be passed if and only if this is > 0.** |
| `surrogateShare` [0.5], `surrogateFrom` [5], `surrogateEpochs` [10] | As in the stage. `surrogateFrom` counts iterations. |
| `surrogateImprovers` [false] | Pre-screen the improvers too |
| `surrogateAim` [false] | Start a retargeted explorer where the model predicts nearest its target |

You may pass your own options type `O` instead. It needs every field above,
with the same names and types, and may have more. An `O` with a
`Ranking ranking` field sets the archive's ranking; without one, the
archive uses `Ranking.init`. `ArchiveOptions` has no `ranking` field, so a
different ranking needs your own `O`. `O` must support `==`, because a
resumed run checks that its options equal the saved ones (all but `evals`).

The restart and explorer rules are compile-time constants in
`solver.archive`, not options: `RESTART_GENERATIONS` (5),
`RESTART_INFEASIBLE` (0.9), `RESTART_SIGMA` (1e-4), `EXPLORE_STALE` (10),
`EXPLORE_PROGRESS` (0.01), `EXPLORE_BLOCK` (30) and `AIM_SAMPLES` (200).

### Callbacks

- `progress(const ref Generation g)` is called after every emitter's
  generation. `ArchiveGeneration` has these fields:
  - `iteration`, `emitter`, `evals`
  - `cells` (coverage), `qdScore`, `bestScore`
  - `improved`, `newCells`, `sigma`
  - `restartReason` (`reasonText` gives its name)
  - `causes`, `explorer`, `target`, `distance`, `focus`
- `afterIteration(const ref Run r)` is called after each iteration except
  the last, with the run exactly as it would be saved. Return true to stop
  there. `run.stopped` is then set.
- `evaluated(const ref Batch b)` gets every batch as soon as it's
  evaluated: the seeds (`iteration == 0`, `emitters` empty), then each
  iteration. Candidate `i` comes from emitter `i / lambda`. Use it to log
  every evaluation.

### Results

`Run` (an `ArchiveRun`) holds the result and everything needed to continue:

- `archive`: an `Archive` with these members:
  - `elites[c]`: `occupied`, `genes` in [0, 1], `eval`, `found`
    (evaluation count when it entered) and `value`
  - `threshold[c]`, `alpha`
  - `coverage()`, `qdScore()`, `best()` (cell index, `size_t.max` when
    empty), `bestScore()`, `occupiedCells()`
- `evals`, `iterations`, `seedEvals`, `lambda`
- `causes`
- `restartsIdle`, `restartsInfeasible`, `restartsConverged`
- `reached` and `abandoned` explorer targets
- `seeds` (the seeds' evaluations)
- `emitters`, `blockedUntil`
- `guide` and `screen`: how the model's picks turned out
- `stopped`

`selectDiverse(archive, grid, within, count)` picks elites to show: the
best first, then repeatedly the elite farthest in bins from those already
picked, among elites at most `within` below the best.

### Focused runs

To spend the rest of a stopped run improving chosen cells:

1. Set `run.focus` to the cells and `run.focusFrom = run.iterations`.
2. Call `refocus(run, k, modelReady)` for each emitter `k`.
3. Resume the run.

Every emitter then improves, restarting from the focused elites in turn.
`focusScores` and `focusImprovers` are bookkeeping for the caller: the
archive saves them but doesn't read them.

---

## A learned model

`solver.model.LearnedModel!L` trains a small neural-network ensemble on
every evaluation of a run, and predicts candidates before they are
evaluated. It is both a `StageModel!(L.Eval)` and an
`ArchiveSurrogate!(L.Eval)`, so you can pass it to `cmaStage` or to
`runArchive`.

### The learning spec

`L` describes what the network learns of your problem: `alias Eval`,
`dimension()`, and the surrogate members of the problem interface.
`isLearning!L` checks it. A problem with those members (`hasSurrogate!P`)
is a learning spec itself, as the worked example's `Beam` is; a separate
struct with them works too, when the features need state the problem
doesn't hold (pleaser's spec keeps its feature map apart from its
evaluator).

```d
struct Beam
{
    // ... the problem's members, then:
    size_t featureCount() const { return 2; }     // network inputs
    void features(const(double)[] genes, double[] f) const { f[] = genes[]; }
    size_t targetCount() const { return 4; }      // network outputs

    void targets(const ref Eval e, double[] t) const
    {
        t[0] = e.scored ? e.objective : double.nan;     // NaN: not learned from this row
        t[1] = e.breach.hard + e.breach.terminal;
        t[2] = e.width;
        t[3] = e.rate;
    }

    Predicted predicted(const(double)[] m) const
    {
        // 0.5 of headroom on a prediction.
        return Predicted(m[0], Breach(m[1] > 0.5 ? m[1] : 0), m[2 .. 4].dup);
    }
}
```

- `features` maps genes (clamped to [0, 1]) to the network's inputs. It's
  called **from several threads at once** when the model has a pool, so it
  must not write shared state.
- `targets` gives one value per output. NaN, or any value that isn't
  finite, counts as missing and adds nothing to the loss. An unscored
  objective is the usual case.
- `predicted` turns outputs into a `Predicted`: an objective, a `Breach`
  and the descriptors (as many as the problem's `descriptorCount`; the
  archive asks for them, a stage ignores them). It's also called from
  several threads at once. Add headroom here to make predictions err
  towards infeasible.

### Using it

```d
alias Model = LearnedModel!Beam;
auto model = new Model(Beam(), seed, pool);   // pool: a TaskPool, or null

// A pre-screened stage:
StageOptions so = {lambda: 8, sigma: 0.2, budget: 800, seed: 42, surrogatePool: 4};
cmaStage(p, best, x0, so, model);

// Model-guided explorers, pre-screened improvers, model-aimed explorers:
ArchiveOptions ao = {..., surrogatePool: 4, surrogateFrom: 3, surrogateEpochs: 5,
                     surrogateImprovers: true, surrogateAim: true};
A.runArchive(p, grid, seeds, ao, null, null, null, null, model);
```

The constructor standardises the inputs on the features of 4000 uniformly
random gene vectors drawn from `seed`, so `features` must accept any genes
in [0, 1]. The ensemble uses `EnsembleOptions` defaults (5 members, two
hidden layers of 64, Adam at 1e-3) with your seed, unless you pass your
own as a fourth argument (its `seed` is replaced by the model's):
`new Model(Beam(), seed, pool, EnsembleOptions(3, 32))`.

Other members:

- `rows()` is the number of evaluations learned.
- `predictEach(genes, take)` gives raw outputs per candidate, for your own
  studies.
- `trainSeconds`, `predictSeconds` and `trainings` are timings.
- `save(evals)` and `restore(bytes, evals)`; see the next section.

`columnStats(rows)` is the column mean and population standard deviation
helper the model uses.

---

## Stopping, saving and resuming

### Within one process

Stop a run with `afterIteration`, then pass it back as `resume`. Use the
same problem, grid, options (`evals` may be larger) and model:

```d
auto run = A.runArchive(p, grid, seeds, ao, null, null, (const ref A.Run r) => r.iterations == 10);
A.runArchive(p, grid, null, ao, null, &run);     // changes run in place, returns a copy
```

A run stopped and resumed ends exactly as the run that never stopped. A
finished run resumed with a larger `evals` ends as a run given that budget
from the start. A resumed run checks three things, and throws if any
fails:

- its options equal the saved ones (all but `evals`);
- it still has its emitters;
- a model is given when its emitters sample for one.

### To a file

solver gives you the state as plain data and leaves **the file format to
you**. It has no serialiser, by design (see [gap 1](#gaps-and-caveats)):
a file format is a project's own choice. This section is a template to
copy into your project and adapt instead, the code the worked example
runs.

#### What to save

| What | From | Back with |
|---|---|---|
| A fingerprint of what the run was computed from | your problem | check with `fingerprintMismatch` |
| The run's state between iterations | `runState(run)` → `RunState!O` | `restoreRun(run, state)` |
| The archive: elites, **thresholds and alpha** | `run.archive` | assign `run.archive` |
| The cause counts | `run.causes` | assign `run.causes` |
| The model, if any | `model.save(run.evals)` → `ubyte[]` | `model.restore(bytes, run.evals)` |

`RunState!O` holds every field of the run except the archive, the causes
and the seeds' evaluations. Each emitter's CMA-ES is a `CmaEs.State`, and
its RNG is a `ulong`. `restoreRun` doesn't touch `run.archive`, so restore
the archive yourself, thresholds and `alpha` included. A resumed run
doesn't reset `alpha` from the options. `LearnedModel.save` is versioned
and holds the training data and the ensemble. `restore` refuses bytes
from another run size or shape, and leaves the model as it was when it
does.

The **fingerprint** is name/value pairs describing what the run was
computed from, such as problem inputs, settings and versions. Check it on
load with `solver.problem.fingerprintMismatch(saved, now)`. It returns
null when every saved pair is in `now` with the same value, in any order,
or the first that isn't as a message. Pairs only in `now` are new and
accepted, so a project can add a setting to its fingerprint without
refusing older files; when a new setting changes results, give it a
default value its older files can be read as, or check it yourself.

#### The format

A file is a header, then one value:

- **Header**: your project's magic string (so another file is refused
  outright) and a format version (`uint`).
- **A struct**: its field count, then each field as its name, its byte
  length and its bytes. The reader matches fields **by name**, so their
  order doesn't matter, and the lengths let it step over a field.
- **A static array** or **a slice**: its length, then its elements.
- **A scalar**: its bytes, through `solver.bytes.Bytes` (native layout, so
  little-endian on little-endian machines, and a `size_t` field is as wide
  as the build's). Lengths are `ulong`.

Writing a struct straight through `Bytes.put` is wrong for any struct
holding arrays, because it writes their pointers. The template walks the
fields instead, and refuses at compile time a type holding a pointer, a
class, a delegate or an associative array.

#### The template

Copy this module into your project as it is; adapt `MAGIC`, `FORMAT` and
`Saved` (below) rather than the module. It is
[`docs/example/source/checkpoint.d`](example/source/checkpoint.d), and the
example checks at compile time that the two are the same text.

<!-- checkpoint.d -->
```d
/// A checkpoint file format to copy into a project and adapt (the developer
/// manual's "Stopping, saving and resuming → To a file"). solver gives a
/// run's state as plain data (RunState, the archive, the causes) and the
/// model's as bytes, and leaves the file format to the caller.
///
/// A file is a header (the project's magic string and its format version)
/// and one value. A struct is written as its named fields, each with its
/// byte length, so the reader matches fields by name: a field the file
/// lacks keeps the default the reader starts from (older files stay
/// readable when a field is added), and a field the reader doesn't know is
/// refused (a newer file isn't half read) unless it is listed as retired.
/// A static array is its elements, a slice its length then its elements, a
/// scalar its bytes; lengths are ulong. Everything goes through
/// solver.bytes, so the bytes are little-endian on little-endian machines.
module checkpoint;

import std.exception : enforce;
import std.traits : isPointer, isAssociativeArray, isDelegate, isFunctionPointer;

import solver.bytes : Bytes;

/// Appends the header: `magic`, then `version_`.
void writeHeader(ref Bytes o, string magic, uint version_)
{
    write(o, magic);
    o.put(version_);
}

/// Reads the header `writeHeader` wrote and returns its version. Throws
/// when the magic isn't `magic` (not this project's file) or the version
/// is above `current` (a newer program's file).
uint readHeader(ref Bytes r, string magic, uint current)
{
    string m;
    read(r, m);
    enforce(m == magic, "not a " ~ magic ~ " file");
    immutable uint v = r.get!uint;
    enforce(v <= current, "the file is of a newer format than this program reads");
    return v;
}

/// Appends `v` to `o`: a struct its fields by name, a static array element
/// by element, a slice its length then its elements, a scalar its bytes.
void write(T)(ref Bytes o, const ref T v)
{
    static assert(!isPointer!T && !isAssociativeArray!T && !isDelegate!T && !isFunctionPointer!T
                  && !is(T == class) && !is(T == interface) && !is(T == union),
                  T.stringof ~ " holds a reference; save what it refers to instead");
    static if (is(T == struct))
    {
        o.put(ulong(v.tupleof.length));
        static foreach (i; 0 .. v.tupleof.length)
        {{
            immutable string name = __traits(identifier, T.tupleof[i]);
            write(o, name);
            // The field's byte length, filled in once it is written.
            immutable size_t at = o.data.length;
            o.put(ulong(0));
            write(o, v.tupleof[i]);
            immutable ulong n = o.data.length - at - ulong.sizeof;
            o.data[at .. at + ulong.sizeof] = (cast(const(ubyte)*) &n)[0 .. ulong.sizeof];
        }}
    }
    else static if (is(T == E[n], E, size_t n))
    {
        o.put(ulong(n));
        foreach (ref e; v)
            write(o, e);
    }
    else static if (is(T == U[], U))
    {
        o.put(ulong(v.length));
        foreach (ref e; v)
            write(o, e);
    }
    else
        o.put(cast() v);
}

/// Reads into `v` what `write` wrote. A struct field the file lacks keeps
/// the value `v` has, so start from a default (`T v;`). A field in the file
/// that `T` has no longer is skipped when `retired` names it as
/// "Struct.field" (`Struct` the type's name without template arguments),
/// and throws otherwise.
void read(T)(ref Bytes r, ref T v, const(string)[] retired = null)
{
    static if (is(T == struct))
    {
        immutable ulong fields = r.get!ulong;
        foreach (_; 0 .. fields)
        {
            string name;
            read(r, name);
            immutable ulong n = r.get!ulong;
            enforce(n <= r.data.length - r.at, "checkpoint cut short");
            auto field = Bytes(r.data[r.at .. r.at + cast(size_t) n]);
            r.at += cast(size_t) n;
            bool known;
            static foreach (i; 0 .. v.tupleof.length)
                if (name == __traits(identifier, T.tupleof[i]))
                {
                    read(field, v.tupleof[i], retired);
                    enforce(field.done, "checkpoint field " ~ name ~ " of another type");
                    known = true;
                }
            if (!known)
            {
                import std.algorithm : canFind;
                enforce(retired.canFind(__traits(identifier, T) ~ "." ~ name),
                        "checkpoint field " ~ __traits(identifier, T) ~ "." ~ name ~ " unknown to this program");
            }
        }
    }
    else static if (is(T == E[n], E, size_t n))
    {
        enforce(r.get!ulong == n, "checkpoint array of another length");
        foreach (ref e; v)
            read(r, e, retired);
    }
    else static if (is(T == immutable(char)[]))
    {
        char[] s;
        read(r, s);
        v = s.idup;
    }
    else static if (is(T == U[], U))
    {
        immutable ulong n = r.get!ulong;
        enforce(n <= r.data.length - r.at, "checkpoint cut short");
        v = new U[cast(size_t) n];
        foreach (ref e; v)
            read(r, e, retired);
    }
    else
        v = r.get!T;
}

unittest
{
    import std.exception : assertThrown;

    // Values come back as they went in, whatever they nest.
    static struct S { double a; int[2] b; string s; double[][] m; bool f; }
    const S x = {1.5, [3, 4], "beam", [[1.0], [], [2.0, 3.0]], true};
    Bytes o;
    writeHeader(o, "test", 1);
    write(o, x);
    auto r = Bytes(o.data);
    assert(readHeader(r, "test", 1) == 1);
    S y;
    read(r, y);
    assert(y == x && r.done);

    // Another project's file, or a newer format's, is refused.
    auto other = Bytes(o.data);
    assertThrown(readHeader(other, "other", 1));
    Bytes newer;
    writeHeader(newer, "test", 2);
    auto rn = Bytes(newer.data);
    assertThrown(readHeader(rn, "test", 1));

    // A field added since: the older file still reads, the new field keeps
    // its default. Fields are matched by name, so their order may change.
    static struct Added { bool f; string s; double a; int[2] b; double[][] m; int count = 7; }
    Added z;
    auto ra = Bytes(o.data);
    readHeader(ra, "test", 1);
    read(ra, z);
    assert(z.a == 1.5 && z.b == [3, 4] && z.s == "beam" && z.m == x.m && z.f && z.count == 7 && ra.done);

    // A field dropped since: refused unless retired.
    static struct Dropped { double a; int[2] b; string s; bool f; }
    Dropped d;
    auto rd = Bytes(o.data);
    readHeader(rd, "test", 1);
    assertThrown(read(rd, d));
    rd = Bytes(o.data);
    readHeader(rd, "test", 1);
    read(rd, d, ["Dropped.m"]);
    assert(d.a == 1.5 && d.s == "beam" && d.f && rd.done);

    // A field whose type changed is refused.
    static struct Changed { double a; int[2] b; string s; double[][] m; double f; }
    Changed c;
    auto rc = Bytes(o.data);
    readHeader(rc, "test", 1);
    assertThrown(read(rc, c));
}
```

The worked example uses it like this: one struct, `Saved`, holds
everything a checkpoint holds, so every part of the file is a named field.

```d
/// The checkpoint file's magic string and format version. Bump the
/// version when a saved field changes meaning or type, and teach load to
/// read the older one; adding a field needs no bump (see checkpoint.d).
enum string MAGIC = "solver-example checkpoint";
enum uint FORMAT = 1;

/// What a checkpoint holds: the problem's fingerprint, the run's state, its
/// archive and causes, and the model.
struct Saved
{
    string[2][] fingerprint;
    RunState!ArchiveOptions state;
    typeof(A.Run.archive) archive;
    A.Run.Causes causes;
    ubyte[] model;
}

/// Saves a run stopped between iterations.
void save(string path, const ref Beam p, const ref A.Run run, Model model)
{
    const Saved s = {fingerprint: p.fingerprint, state: runState(run), archive: run.archive, causes: run.causes,
                     model: model.save(run.evals)};
    Bytes o;
    checkpoint.writeHeader(o, MAGIC, FORMAT);
    checkpoint.write(o, s);
    write(path, o.data);
}

/// Loads what save wrote into a new run, ready to pass as runArchive's
/// `resume`, and restores the model.
A.Run load(string path, const ref Beam p, Model model)
{
    auto r = Bytes(cast(ubyte[]) read(path));
    checkpoint.readHeader(r, MAGIC, FORMAT);
    Saved s;
    checkpoint.read(r, s);
    if (!r.done)
        throw new Exception("the checkpoint has bytes left over");
    if (const why = fingerprintMismatch(s.fingerprint, p.fingerprint))
        throw new Exception("the checkpoint is of another problem: " ~ why);
    A.Run run;
    restoreRun(run, s.state);
    run.archive = s.archive;
    run.causes = s.causes;
    model.restore(s.model, run.evals);
    return run;
}
```

The example then checks that the resumed run's archive is the
uninterrupted run's, bit for bit.

#### Adapting it

- **Name the file**: give `MAGIC` your project's name, and put what a
  checkpoint holds in `Saved`, one field per thing to save. Anything your
  problem keeps between iterations (its own counters, a schedule) goes in
  `Saved` too, as plain data.
- **Adding a field** (to `Saved`, or when a solver upgrade adds one to
  `RunState`, `Archive` or `ArchiveOptions`) needs no new version. An
  older file lacks it, and the field keeps the default it has in the
  `Saved s;` the reader starts from. Give a field you add a default
  initialiser that means "as before" (pleaser does the same in its text
  formats, below). When no default can mean that, bump `FORMAT`.
- **Removing a field**: an older file still has it, and `read` refuses
  any field it doesn't know, so that a newer program's file is never half
  read. Name the removed field in `read`'s `retired` list, as
  `"Struct.field"` (the struct's name without template arguments):
  `checkpoint.read(r, s, ["Saved.schedule"]);`.
- **Changing a field's type or meaning, or renaming it**: bump `FORMAT`
  and convert in `load`. Keep the old layout as a struct of its own and
  read it when the header says so:

  ```d
  enum uint FORMAT = 2;            // for example, 2: the fingerprint became a struct

  struct SavedV1 { string[2][] fingerprint; RunState!ArchiveOptions state; /* ... */ ubyte[] model; }

  A.Run load(string path, const ref Beam p, Model model)
  {
      auto r = Bytes(cast(ubyte[]) read(path));
      Saved s;
      if (checkpoint.readHeader(r, MAGIC, FORMAT) == 1)
      {
          SavedV1 old;
          checkpoint.read(r, old);
          s = upgrade(old);        // your conversion
      }
      else
          checkpoint.read(r, s);
      // ... as before
  }
  ```

  A field whose type changed without a bump is refused when its bytes
  don't fit the new type (`"of another type"`). One of the same size
  (`int` to `float`) isn't caught: bump the version.
- **solver upgrades**: a field solver adds reads as its default, and
  nothing promises that default means "as before". A resumed run is only
  guaranteed to be the uninterrupted run, bit for bit, on the solver it
  was saved with. If that matters, put your solver revision in the
  fingerprint.
- **Platforms**: a checkpoint loads on any platform, but resumes bit for
  bit only on the one it was saved on; elsewhere it goes on as that
  platform's run would (runs differ between platforms). Record the
  platform in the file (a field of `Saved`, or the fingerprint's
  information) if you need to tell; pleaser's states name theirs.
- **Write through a temporary file and a rename**, so a stop in mid-write
  leaves the last checkpoint whole: `write(path ~ ".tmp", o.data);
  rename(path ~ ".tmp", path);`.

pleaser's `elites.state` (`elitesText` and `parseElites` in
`pleaser/source/pleaser/archive.d`) is a real versioned format, as text: a
header line, then one tagged line per thing (`check`, `alpha`,
`threshold`, `elite`, `run ...`), with numbers in hexadecimal floating
point so they read back bit for bit. Fields added since its first version
are written only when they differ from their default (`cost` on an elite
line, the model-guided explorers' `run surrogate` line), so older files
read as before and files of runs without them are unchanged. A text
format costs more code per field and is worth it when people read or edit
the files; the template above is the shorter route.

A stage has no resume. `CmaEs` does have `save` and `restore` (see below)
if you write your own loop.

---

## Lower-level pieces

- **`solver.cmaes.CmaEs`**: deterministic CMA-ES for minimisation.
  - `CmaEs(x0, sigma0, lambda, seed)`, where `lambda = 0` gives
    4 + ⌊3 ln n⌋.
  - `ask()` returns `lambda` rows, valid until the next `ask`.
    `tell(fitness)` takes their fitness.
  - `save()` returns a `CmaEs.State`. `CmaEs.restore(state)` continues
    bit-identically when the state was taken between `tell` and `ask`.
  - Accessors: `stepSize`, `currentMean`, `populationSize`, `generation`,
    `bestFitness`, `bestPoint`.
  - `jacobiEigen` is its eigensolver.
- **`solver.rng.SplitMix64`**: the only RNG solver uses, and the one your
  problem should use for any seeded noise (see
  [Determinism](#determinism)).
  - `next()`, `uniform01()` (53 bits), `bipolar()`.
  - Its state is the public `ulong state`.
  - It is the same generator as pleaser's `pleaser.rng`, bit for bit.
- **`solver.detmath`**: `log2` and `exp2` that give the same bits on every
  platform, for a problem whose results must match across platforms.
  - `log2d(x)` for finite `x > 0` (no denormals or zero: floor the input
    first), `exp2d(y)` for `y` in [−1000, 1000], exactly 1 at `y = 0`.
    About 1e-15 relative error.
  - `log2v` and `exp2v` do the same on a 4-lane `__vector(double[4])`
    (`V`); every lane equals the scalar result bit for bit, so a loop can
    run vectors with a scalar tail. `select(mask, a, b)` is the lane-wise
    `mask ? a : b`.
  - Plain IEEE arithmetic, no `libm`. Its unittests hold it to reference
    bit patterns, and CI runs them on Linux and Windows.
  - It makes the maths it is used for platform-independent, not a whole
    run: solver itself still uses `libm` (see
    [Determinism](#determinism), item 5).
  - pleaser and its plugins keep their own identical copy,
    `pleaser.detmath`.
- **`solver.ensemble.Ensemble`**: the network ensemble under
  `LearnedModel`, usable on its own.
  - `new Ensemble(inputs, outputs, EnsembleOptions)`
  - `scaleInputs(mean, sd)`
  - `train(x, y, epochs, pool)`, with NaN targets treated as missing
  - `predict(x, mean, spread)`
  - `save()` and `restore(bytes)`
  - `targetStats` and `Stats`
- **`solver.bytes.Bytes`**: little-endian put and get of scalars and
  arrays, with an array's length first. `getArray!T(expect)` checks the
  length, and `done()` is true when everything has been read. It writes
  native layout with no endianness conversion, so the bytes are
  little-endian only on little-endian machines.
- **`solver.stage.pickRendered`, `stalled`**: the pre-screening pick and
  the stall test, exposed for custom loops.
- **`solver.archive`'s smaller helpers**:
  - `emitterSeed`, `explorerGain`, `predictedFitness`
  - `nearlyAllInfeasible`, `explorerRestart`, `improverRestart`
  - `pickTarget`, `refocus`, `copyOf`
  - `NoCauses`, `CausesOf!P`

---

## Determinism

solver keeps these rules itself:

- Every random draw comes from `SplitMix64`, seeded from the options.
  Emitter `k` draws from its own seed.
- Every population is evaluated in one batch, and results are consumed in
  input order.
- Sorts are stable, with ties broken by index.
- Sums run in a fixed order.
- `Ensemble` and `LearnedModel` give the same bits with any `TaskPool`
  size, or none.

Your problem must keep them too:

1. `evaluate` returns the same bits for the same genes whatever the batch,
   the order of the other candidates in it, and the number of threads. If
   you parallelise, compute each candidate independently and write it to
   its own slot.
2. Draw any seeded noise from `solver.rng.SplitMix64`, seeded from the
   genes or your own fixed seed. Don't use `std.random`, whose algorithms
   aren't guaranteed to stay the same across Phobos versions.
3. Keep floating-point reductions in a fixed order. `std.algorithm.sum`
   sums pairwise and changes the bits. Don't use `@fastmath`. To vectorise
   a reduction under strict IEEE, split it into lanes with independent
   accumulators and combine them in a fixed order.
4. `features`, `targets` and `predicted` in a learning spec must be pure
   functions of their arguments.
5. Runs are the same on any thread count and CPU of one platform, but
   **not across platforms**. LDC turns `exp`, `log`, `pow`, `sin`, `cos`
   and the like into calls to the C maths library, and glibc and the MSVC
   runtime can round the last bit differently. solver itself calls it
   (CMA-ES's normal samples and step size, the explorers' noise,
   log-scaled genes), and so does almost any objective. A last-bit
   difference flips a ranking sooner or later, and from there the runs
   take different paths: pleaser's Linux and Windows runs part within a
   few generations. So:
   - keep golden runs and reference outputs **per platform**, and compare
     across platforms by results over seeds, not run for run;
   - a test that needs something to happen during a run (an explorer
     aimed, a restart) must force it, so it happens on every platform;
   - a saved run resumes bit for bit on the platform it was saved on. On
     another it resumes, but goes on as that platform's run would.

   `solver.detmath`'s `log2d`/`exp2d` (and their 4-lane forms) give the
   same bits everywhere. Use them where a problem's own result must not
   depend on the platform, say a value it stores or compares against a
   fixed reference; they don't make a whole run platform-independent. With
   LDC 1.42.0, plain arithmetic gives the same bits with or without FMA
   instructions (`detmath`'s reference bits are the same built with
   `-mcpu=native` or `-mattr=+fma`); don't add flags such as
   `-fp-contract=fast` or `@fastmath` that change that.

---

## Performance

- **Inlining**: solver is a source library and the searches are templates
  over your problem, so `objective`, `breach` and `descriptors` inline.
  Keep them cheap and `const`.
- **Threads**: solver never starts threads of its own. Parallelism comes
  from your `evaluate` and from the `TaskPool` you give `LearnedModel`.
  Make `lambda × emitters` a multiple of your thread count, so each archive
  batch fills the pool.
- **Stack size**: with `N > 0`, `Archive` holds `Elite!E[N]` by value, and
  `ArchiveRun` holds the archive. Each `Run` local, or a `Run` returned by
  value, takes `N × (sizeof(E) + about 40)` bytes of its caller's frame.
  pleaser's 75-cell archive is about 40 KB, and pleaser needs an 8 MB stack
  on Windows. With `N = 0` (the default) the cells live on the heap, and a
  `Run` is a few hundred bytes. Use `N > 0` only when you have measured a
  gain, and then keep runs off the stack, for example `auto run = new
  A.Run;`.
- **Allocation**: each generation allocates the batch's results and the
  clamped genes. That costs nothing next to a real evaluation. For a
  microsecond-scale problem, allocation can dominate.

---

## Gaps and caveats

These are true of solver main on 2026-10-29. Each
one says what it costs you and how to work around it today.

1. **No checkpoint API, by design.** solver has `RunState`,
   `CmaEs.State`, `LearnedModel.save` and `Bytes`, and nothing that writes
   a run to a file. The intended route is the template in
   [To a file](#to-a-file): copy `checkpoint.d` into your project and
   adapt it. It versions the file and stays readable when a field is
   added. `isSaveable`'s `saveEval`/`loadEval` are a convention for a
   project's own format; nothing in solver calls them.
2. **The terminal tier on stage pre-screening.** `StageModel.predictScore`
   returns one violation, which the stage ranks as hard. `LearnedModel`
   reports `hard + terminal` there, so a predicted terminal breach ranks as
   a hard one in a pre-screened stage. The archive's model interface keeps
   the tiers.
3. **The archive's ranking needs a custom options type.** `ArchiveOptions`
   has no `ranking` field, so an objective outside about −50 to +50 needs
   your own `O` with one (see [Ranking](#ranking)).
4. **Restart and explorer rules are constants.** `RESTART_*`, `EXPLORE_*`
   and `AIM_SAMPLES` can't be changed per run.
5. **No stage resume.** A stage runs to its end. Use the archive, or your
   own loop over `CmaEs`, if you need checkpoints within a stage.
6. **LDC only, and no dub registry package.** See
   [Adding solver to a project](#adding-solver-to-a-project).
7. **Runs differ between platforms.** solver's own CMA-ES and explorers
   use the C maths library, whose last bits differ between glibc and the
   MSVC runtime, so the same run takes different paths on Linux and
   Windows. This is accepted, not planned to change: keep goldens per
   platform (see [Determinism](#determinism)). `solver.detmath` has
   platform-identical `log2` and `exp2` for a problem's own use.
8. **Evaluation log, survey and model test tools stay in pleaser.**
   pleaser's `evals.bin` log, random survey and offline surrogate tests
   are mix-specific and stay there. Use the `evaluated` callback to build
   your own log.

---

## Module reference

| Module | What it holds |
|---|---|
| `solver.problem` | `Tier`, `Breach`, `ConstraintSpec`, `breachOf`, `value`, `Ranking`, `Scoring`, `clamp01`, `boundExcess`, `Scale`, `GeneSpec`, `Predicted`, `isProblem`, `hasDescriptors`, `hasSurrogate`, `isSaveable`, `valueOf`, `fitnessOf`, `isGrid`, `Axis`, `BinnedGrid`, `Outcome`, `DynamicProblem`, `fingerprintMismatch` |
| `solver.stage` | `StageModel`, `StageOptions`, `Best`, `hasCauses`, `StageGeneration`, `StageEnd`, `cmaStage`, `pickRendered`, `stalled` |
| `solver.archive` | `Elite`, `Offer`, `Archive`, `copyOf`, `selectDiverse`, `ArchiveOptions`, `ArchiveSurrogate`, `GuideStats`, `ScreenStats`, `RestartReason`, `reasonText`, `NoCauses`, `CausesOf`, `ArchiveGeneration`, `Emitter`, `ArchiveRun`, `RunState`, `runState`, `restoreRun`, `ArchiveBatch`, `NoFeasibleSeed`, `Archiving` (`runArchive`), the restart constants and helpers |
| `solver.model` | `isLearning`, `LearnedModel`, `ColumnStats`, `columnStats` |
| `solver.cmaes` | `CmaEs` (with `State`), `jacobiEigen` |
| `solver.ensemble` | `EnsembleOptions`, `Ensemble`, `Stats`, `targetStats` |
| `solver.bytes` | `Bytes` |
| `solver.rng` | `SplitMix64` |
| `solver.detmath` | `log2d`, `exp2d`, `log2v`, `exp2v`, `select`, `V`, `VL` |
