# Next session: solver follow-up work

This is a self-contained prompt for a fresh Claude Code session. Read it all
before starting. The work it lists is proposed, not yet approved: ask Dev
which items to take on (and in what order) in your first reply, before
changing anything.

## Where things stand (2026-10-29)

solver is the deterministic black-box optimisation core extracted from
[pleaser](https://github.com/x21-dev/pleaser): CMA-ES stages and a
quality-diversity archive (CMA-MAE) over a generic problem interface, plus a
learned surrogate model. It is a dub **source library** (`targetType:
sourceLibrary`), LDC only (developed with LDC 1.42.0), no dependencies beyond
Phobos and `ldc.simd`.

Merged so far:

| Step | solver | pleaser |
|---|---|---|
| CMA-ES and its RNG moved; pleaser's inlining kept | #1, #2 | #74 |
| Generic CMA-ES stage | #3 | #75 |
| Surrogate ensemble and `Bytes` | #4 | #76 |
| The archive (CMA-MAE over descriptors on a grid) | #5 | #77 |
| Learned model, archive run state, fingerprint check | #6 | #78 |
| Developer manual and worked example | #7 | |
| Model over the problem's surrogate members, one fingerprint check, checked stage options | #8 | #79 |
| CI on Linux and Windows | #10 | |
| Save format as documentation: a versioned checkpoint template | #11 | |
| `solver.detmath`; runs differ between platforms, goldens per platform | #12 | |
| Next-session prompt rewritten | #13 | |
| Housekeeping: CLAUDE.md, SessionStart hook, `DynamicProblem`'s comments, the manual's introduction | #14 | |
| Stages that stop and resume (`StageRun`); cma1 resume in pleaser (`--resume cma1.state`) | #15 | #80 |
| Manual: setting the ranking's offsets | #16 | |
| `ranking` and the restart and explorer rules (`ArchiveRules`) as archive options | PRNUM | |

Every step that touched pleaser was verified with **byte-identical golden
runs** (the same pleaser run before and after, byte-comparing
`elites.state`, `progress.csv` and `best.txt`) and timings within noise.
#10 to #14 changed nothing pleaser runs (pleaser's build, `dub test` and
`./pleaser selftest` were run against #12 to show it). Stage resume did,
and its golden runs are byte-identical apart from `cma1.state`'s new
`stage` lines.

Layout of solver:

- `source/solver/`: `problem.d` (problem contract, tiers, ranking, genes,
  grids, `fingerprintMismatch`), `stage.d` (`cmaStage`), `archive.d`
  (`runArchive`, `RunState`, `runState`/`restoreRun`), `model.d`
  (`LearnedModel`), `ensemble.d`, `cmaes.d`, `bytes.d`, `rng.d`
  (`SplitMix64`), `detmath.d` (`log2d`/`exp2d`, scalar and 4-lane).
- `docs/manual.md`: the developer manual. Its "Gaps and caveats" section
  lists what solver doesn't do yet; the proposals below come from it.
- `docs/example/`: the worked example. `app.d` runs a stage and an archive,
  saves the archive part way through `checkpoint.d`, resumes and checks
  the result against the uninterrupted run bit for bit. `checkpoint.d` is
  the manual's checkpoint template, verbatim: a `static assert` in `app.d`
  fails the build when the manual's copy and the file differ, so change
  both together.
- `.github/workflows/ci.yml`: on pull requests and pushes to `main`, on
  `ubuntu-latest` and `windows-latest` with LDC 1.42.0: `dub test` in
  solver, `dub test` and `dub run --build=release` in `docs/example`.
- `CLAUDE.md`: how to work here (checks, pleaser, determinism, D style).
- `.claude/hooks/session-start.sh`: installs LDC 1.42.0 in cloud sessions
  opened on solver, so `dub test` works without opening the session on
  pleaser.

Related repositories, checked out side by side as `../pleaser`,
`../satisfaction`, `../solver`:

- **pleaser** depends on solver by path (`"solver": { "path": "../solver" }`).
  Its Windows build links with `/STACK:8388608` because its 75-cell
  `Archive` lives by value on the stack. pleaser's
  `.claude/hooks/session-start.sh` clones satisfaction and solver and
  installs LDC 1.42.0 and the dub packages (satisfaction and solver must be
  attached to the session for the clones). Open the session on pleaser, or
  run that hook, when you need to build pleaser. Read `pleaser/CLAUDE.md`
  before changing anything there. pleaser has no CI.
- **satisfaction** (the stimulation analyser) is pleaser's objective
  function and the gold standard for it. It is **not** a solver consumer and
  is never changed from pleaser or solver.

## Decisions already made

Don't reopen these without Dev:

- **Runs differ between platforms, and that is accepted.** LDC turns
  `exp`, `log`, `pow`, `sin`, `cos` into calls to the C maths library,
  whose last bits differ between glibc and the MSVC runtime. solver's own
  CMA-ES (normal samples, step size), the explorers' noise and log-scaled
  genes use it, and so do pleaser's render and satisfaction, so the same
  run takes different paths on Linux and Windows. Golden runs and reference
  outputs are kept **per platform**; a test that needs an event during a
  run must force it. Making whole runs platform-identical (replacing the C
  maths functions process-wide, which would also shift satisfaction's
  results) was sized at about a week and declined.
- **No checkpoint API.** The save format is documentation: the manual's
  template (`docs/example/source/checkpoint.d`), copied and adapted by each
  project. pleaser keeps its own text formats (`elites.state`:
  `elitesText`/`parseElites` in `pleaser/source/pleaser/archive.d`).
- **pleaser and its plugins keep their own `pleaser.detmath` and
  `pleaser.rng`.** `solver.detmath` and `solver.rng` are identical copies;
  don't switch pleaser's code to them.

Parked by Dev (2026-10-29); raise only if something changes:

- **A ranking safety check** (throw when a value reaches the offsets): not
  much value for now; the manual's "Setting the offsets" covers it.
- **CI for pleaser**: GitHub Actions' costs and limits outweigh it for a
  project with little development left. The selftest runs on the GCP
  runner (Linux) before pushing, and Dev runs it on Windows now and then.
- **Resuming a cma1 or archive run on another platform**: not needed.
  Dev resumed a 480-gene cma1 on Windows past 9000 evaluations and is
  happy with it.

## Proposed next steps

In the order I'd suggest. Each is one branch and one PR to `main` on
x21-dev/solver, plus a matching pleaser PR only where pleaser has to
change.

### 1. Gap 2: the terminal tier through stage pre-screening

`StageModel.predictScore` returns one violation, ranked as hard, and
`LearnedModel` reports `hard + terminal` there, so a predicted terminal
breach ranks as a hard one in a pre-screened stage (the archive's model
interface keeps the tiers). Fixing it changes the `StageModel` interface,
which pleaser implements (`CandidateModel = StageModel!Evaluation` in
`pleaser/source/pleaser/optimize.d`): a pleaser PR too. pleaser appears to
have no terminal-tier limits, so its runs should stay byte-identical;
prove it with golden runs.

### Not proposed

The manual's gaps were renumbered as they closed (the C API and Python
mentions, stage resume, the archive's ranking and its rules); the numbers
here are the current ones. A C API or Python binding would be new work to
scope with Dev.

- **Gap 3, a dub registry package**: solver's licence is proprietary, so
  publishing is Dev's call, not a task.
- **Gap 5**: pleaser's evaluation log, survey and model-test tools are
  mix-specific and stay in pleaser.

## How to work

- One branch and one PR per item, on x21-dev/solver (and pleaser only if
  it has to change).
- Before pushing a solver change: `dub test --compiler=ldc2` in solver,
  and in `docs/example` both `dub test --compiler=ldc2` and
  `dub run --compiler=ldc2 --build=release`. CI runs the same on Linux and
  Windows; a PR is done when both are green.
- **Any change that touches what pleaser runs** (solver code pleaser
  imports, or pleaser itself) needs, before and after:
  - pleaser's own checks from `pleaser/CLAUDE.md`:
    `dub build --compiler=ldc2 --build=release-native`,
    `dub test --compiler=ldc2`, `./pleaser selftest`;
  - a golden run: the same `pleaser optimize` run (same stems, seed,
    settings) on `main` and on the branch, on the same platform,
    byte-comparing `elites.state`, `progress.csv` and `best.txt`. They
    must be identical;
  - timings within noise (`./pleaser bench`, or the golden run's wall
    time, a few repeats each side).
  Report both in the PR. The selftest takes longer than 10 minutes on a
  4-core session container; it and long runs can go to pleaser's GCP
  runner (`pleaser/tools/runner/README.md`, about 10 minutes for build and
  selftest) when `GCP_SA_KEY_B64` is set. Never print or commit that key,
  and stop the VM as soon as the work is done.
- When pleaser builds against a solver branch, keep `../solver` checked
  out on that branch until the build and runs finish: switching it in the
  middle mixes two versions. Use `git worktree` for other solver work in
  the meantime.
- Results must stay deterministic: bit-identical on any thread count of
  one platform. Seeded noise only through `SplitMix64`, never
  `std.random`.
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
