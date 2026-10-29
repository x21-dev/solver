# solver

Deterministic black-box optimisation core extracted from pleaser: CMA-ES
stages and a quality-diversity archive over a problem interface. See
README.md for the layout and docs/manual.md for the developer manual.

## Working here

- LDC only, developed with LDC 1.42.0. In cloud sessions the SessionStart
  hook installs it. solver is a dub source library with no dependencies.
- Before pushing: `dub test --compiler=ldc2` in solver, and in
  `docs/example` both `dub test --compiler=ldc2` and
  `dub run --compiler=ldc2 --build=release` (the example checks a resumed
  run against the uninterrupted one bit for bit). CI runs the same on
  Linux and Windows for every pull request; a PR is done when both are
  green.
- pleaser (`../pleaser`) depends on solver by path and is its main user.
  A change to solver code pleaser imports also needs pleaser's checks
  (`pleaser/CLAUDE.md`), a golden run (the same `pleaser optimize` run on
  `main` and on the branch, on one platform, byte-comparing
  `elites.state`, `progress.csv` and `best.txt`: identical) and timings
  within noise, all reported in the PR. Open the session on pleaser, or
  run its SessionStart hook, to build it. While pleaser builds against a
  solver branch, keep `../solver` on that branch; use `git worktree` for
  other solver work in the meantime.
- Never change satisfaction (`../satisfaction`, pleaser's analyser) from
  here, and don't switch pleaser's own `pleaser.detmath` or `pleaser.rng`
  to solver's copies.
- `docs/example/source/checkpoint.d` is printed whole in the manual (To a
  file) as the checkpoint template. Change both together: a
  `static assert` in `docs/example/source/app.d` fails the build when
  they differ.
- The manual's Gaps and caveats list what solver doesn't do. Keep it, and
  its date, true when a change closes or opens one.
- `docs/next-session.md` is the hand-over prompt for the next session;
  update it when the state or the plan changes.

## Determinism

- Results are bit-identical on any thread count and CPU of one platform.
  Seeded noise only through `solver.rng.SplitMix64`, never `std.random`.
- Runs differ between platforms, and that is accepted: LDC lowers `exp`,
  `log`, `pow`, `sin`, `cos` to the C maths library, whose last bits differ
  between glibc and the MSVC runtime. Keep golden runs and reference
  outputs per platform. A test that needs something to happen during a run
  (an explorer aimed, a restart) must force it, so it happens on Linux and
  Windows alike.
- `solver.detmath` gives the same bits everywhere; its unittests hold it
  to reference bit patterns. Don't add flags that change floating-point
  results (`-fp-contract=fast`, `@fastmath`).

## D style (target LDC)

- Write float selects as `cond ? a : b`, not multiply-by-bool.
- Prefer a functional style wherever it costs neither performance nor
  readability: pure functions (marked `pure`), `const`/`immutable` by
  default, a struct changed only once everything it needs has been
  computed and checked. Not at the cost of vectorisation or allocations in
  hot loops: there keep indexed loops and array operations (`zip`/`lockstep`
  usually stop LDC vectorising).
- Aim for loops LDC can vectorise, and check the asm when it matters. A
  hot loop that writes through one slice and reads a struct's other
  slices: copy them into locals first, or LLVM reloads their pointers every
  iteration and leaves the loop scalar.
- Branchless binary searches need `-x86-cmov-converter=false`.
- No `@fastmath`. Vectorise reductions by splitting into lanes under strict
  IEEE: `enum lanes = 4;` by default, `static foreach` over independent
  accumulators and a fixed-order combine at the end. Floating-point sums
  keep their fixed order: never `std.algorithm.sum` (pairwise; changes the
  bits).
- Biquads: double-precision coefficients and state.
- Keep large values off the stack. With `N > 0` an `Archive` holds its
  cells by value (pleaser's 75-cell one is about 40 KB, and pleaser links
  Windows with an 8 MB stack); Windows' default main-thread stack is 1 MB,
  and an exit with no output and code -1073741571 (0xC00000FD) is a stack
  overflow.
