# Next session: solver follow-up work

This is a self-contained prompt for a fresh Claude Code session. Read it all
before starting, then do the three tasks below in order.

## Where things stand (2026-10-28)

solver is the deterministic black-box optimisation core extracted from
[pleaser](https://github.com/x21-dev/pleaser): CMA-ES stages and a
quality-diversity archive (CMA-MAE) over a generic problem interface, plus a
learned surrogate model. It is a dub **source library** (`targetType:
sourceLibrary`), LDC only (developed with LDC 1.42.0), no dependencies beyond
Phobos and `ldc.simd`.

The extraction is finished and merged:

| Step | solver | pleaser |
|---|---|---|
| CMA-ES and its RNG moved; pleaser's inlining kept | #1, #2 | #74 |
| Generic CMA-ES stage | #3 | #75 |
| Surrogate ensemble and `Bytes` | #4 | #76 |
| The archive (CMA-MAE over descriptors on a grid) | #5 | #77 |
| Learned model, archive run state, fingerprint check | #6 | #78 |
| Developer manual and worked example | #7 | |
| Follow-up fixes: model over the problem's surrogate members, one fingerprint check, checked stage options | #8 (merged) | #79 (merged) |

Every step that touched pleaser was verified with **byte-identical golden
runs** (the same pleaser run before and after the change, byte-comparing
`elites.state`, `progress.csv` and `best.txt`) and timings within noise.

Layout of solver:

- `source/solver/`: `problem.d` (problem contract, tiers, ranking, genes,
  grids, `fingerprintMismatch`), `stage.d` (`cmaStage`), `archive.d`
  (`runArchive`, `RunState`, `runState`/`restoreRun`), `model.d`
  (`LearnedModel`), `ensemble.d`, `cmaes.d`, `bytes.d`, `rng.d`
  (`SplitMix64`).
- `docs/manual.md`: the developer manual. Its "Gaps and caveats" section
  lists what solver doesn't do yet.
- `docs/example/`: the worked example (`app.d`, and `checkpoint.d`, a
  60-line generic struct writer over `solver.bytes.Bytes`).
- No CI, no CLAUDE.md, no SessionStart hook of its own. Tests
  (`dub test --compiler=ldc2`, unittests in every module) have **only been
  run on Linux**.

Related repositories, checked out side by side as `../pleaser`,
`../satisfaction`, `../solver`:

- **pleaser** depends on solver by path (`"solver": { "path": "../solver" }`).
  Its Windows build links with `/STACK:8388608` (`lflags-windows` in its
  `dub.json`) because its 75-cell `Archive` lives by value on the stack.
  pleaser's `.claude/hooks/session-start.sh` clones satisfaction and solver
  and installs LDC 1.42.0 and the dub packages; open the session on pleaser
  (or run that hook) when you need to build pleaser. Read
  `pleaser/CLAUDE.md` before changing anything there.
- **satisfaction** (the stimulation analyser) is pleaser's objective
  function and the gold standard for it. It is **not** a solver consumer and
  is never changed from pleaser or solver.

## The tasks, in order

The developer (Dev) approved this direction. Do them in this order.

### 1. Automated tests (CI) on Linux and Windows

Dev hasn't explicitly confirmed this one. Ask once in your first reply, but
start the work on a branch anyway; it's reversible.

- Add a GitHub Actions workflow to solver (`.github/workflows/`) that, on
  `ubuntu-latest` and `windows-latest`, installs **LDC 1.42.0** (pin it, for
  example with `dlang-community/setup-dlang` and `compiler: ldc-1.42.0`),
  then runs `dub test --compiler=ldc2` and builds and runs the worked
  example (`cd docs/example && dub run --compiler=ldc2 --build=release`).
  The example checks that a resumed run's archive equals the uninterrupted
  run's bit for bit, so it is a test too.
- Expect Windows to surface things Linux never did: the default 1 MB main
  thread stack (an exit with no output and code -1073741571 / 0xC00000FD is
  a stack overflow; see the manual's Performance section on `N > 0`
  archives), path and line-ending differences, and last-bit differences from
  the MSVC runtime's `exp`/`log`. A test that needs an event during a run
  (an explorer aimed, a restart) must force it so it happens on both
  platforms, not rely on a run that happens to produce it on Linux.
- Fix real failures in solver. Don't skip, disable or loosen a test to get
  green. If a fix changes solver code pleaser runs through, it is a
  pleaser-touching change: do the golden-run and timing checks below.

### 2. Copy detmath (and the RNG) into solver

- The RNG is **already in solver**: `solver.rng.SplitMix64` is identical to
  `pleaser.rng.SplitMix64` (it moved in #1). pleaser's archive imports
  `solver.rng`; the rest of pleaser, the x21 Strip plugin and the reverb
  plugin keep `pleaser.rng`. Nothing to copy; just make the manual say
  clearly that `solver.rng` is the RNG a problem should use.
- Copy `pleaser/source/pleaser/detmath.d` to
  `source/solver/detmath.d` as `module solver.detmath`, unchanged apart
  from the module name: `log2d`/`exp2d` (scalar) and `log2v`/`exp2v`
  (4-lane `__vector(double[4])`), every lane bit-identical to the scalar,
  plain IEEE arithmetic so every platform gives the same bits. Bring its
  unittests, and add a test with reference bit patterns so CI checks the
  results are the same on Linux and Windows.
- **pleaser and the plugins keep their own copies** (`pleaser.detmath`,
  used by `dsp.d`, `cost.d`, `surrogate.d`, and by `plugin/` and `strip/`).
  Don't switch them to solver's. So pleaser's code doesn't change; still
  build pleaser and run `dub test` and `./pleaser selftest` against the new
  solver to show nothing broke.
- Update `docs/manual.md`: Determinism item 5, gap 8 ("No platform-identical
  maths") and the module reference table, plus a line in `README.md`.

### 3. Save format as documentation only

No new solver API. Dev's view: a generic checkpoint format belongs in the
documentation as a template and code that developers copy verbatim or
adapt to their project.

- Turn the example's `docs/example/source/checkpoint.d` and the manual's
  "Stopping, saving and resuming → To a file" section into a documented
  template in the manual: the full writer/reader code, what to save (run
  state, archive with thresholds and alpha, causes, model bytes,
  fingerprint), and how to adapt it. Show at least how to add a format
  version header and how to stay readable when a field is added (the
  current writer's format changes whenever any saved type gains a field).
  pleaser's `elites.state` text format (`pleaser/source/pleaser/state.d`)
  is a real example of a versioned format to point at.
- Keep the worked example compiling against the template, and keep its
  bit-for-bit resume check.
- Update gap 1 ("No generic checkpoint format") to say the template is the
  intended route rather than a missing feature.

## How to work

- Each task on its own branch with a PR to `main` on x21-dev/solver (and a
  matching pleaser PR only if pleaser has to change). One PR per task.
- Before pushing a solver change: `dub test --compiler=ldc2` in solver, and
  `cd docs/example && dub run --compiler=ldc2 --build=release`.
- **Any change that touches what pleaser runs** (solver code pleaser
  imports, or pleaser itself) needs, before and after:
  - pleaser's own checks from `pleaser/CLAUDE.md`:
    `dub build --compiler=ldc2 --build=release-native`,
    `dub test --compiler=ldc2`, `./pleaser selftest`;
  - a golden run: the same `pleaser optimize` run (same stems, seed,
    settings) on `main` and on the branch, byte-comparing `elites.state`,
    `progress.csv` and `best.txt`. They must be identical;
  - timings within noise (`./pleaser bench`, or the golden run's wall
    time, a few repeats each side).
  Report both in the PR. Long runs may go to pleaser's GCP runner
  (`pleaser/tools/runner/README.md`) when `GCP_SA_KEY_B64` is set; never
  print or commit that key, and stop the VM as soon as the work is done.
- Results must stay deterministic: bit-identical on any thread count. Seeded
  noise only through `SplitMix64`, never `std.random`.
- Never change satisfaction.

## D style (target LDC)

Dev's preferences, the same as `pleaser/CLAUDE.md`:

- Write float selects as `cond ? a : b`, not multiply-by-bool.
- Aim for loops LDC can vectorise, and check the asm when it matters. A hot
  loop that writes through one slice and reads a struct's other slices:
  copy them into locals first, or LLVM reloads their pointers each
  iteration and leaves the loop scalar. `zip`/`lockstep` usually stop LDC
  vectorising.
- Branchless binary searches need `-x86-cmov-converter=false`, or LLVM
  turns the cmov back into a jump.
- No `@fastmath`. Vectorise reductions by splitting into lanes under strict
  IEEE: `enum lanes = 4;` by default, `static foreach` over independent
  accumulators and a fixed-order combine at the end. More lanes only if
  significantly faster. Never `std.algorithm.sum` (pairwise; changes the
  bits).
- Biquads: double-precision coefficients and state.
- Prefer a functional style (pure functions, `const`/`immutable` by
  default) where it costs neither performance nor readability.
- Keep large values off the stack.
