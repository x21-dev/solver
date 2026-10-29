/// The developer manual's worked example (docs/manual.md): a constrained
/// problem through a CMA-ES stage, then through the archive with a learned
/// model, saved to a file part way, resumed from it, and checked against
/// the run that never stopped.
module app;

import std.exception : enforce;
import std.file : read, remove, tempDir, write;
import std.path : buildPath;
import std.stdio : writefln;

static import checkpoint;
import solver.archive;
import solver.bytes : Bytes;
import solver.model : LearnedModel, isLearning;
import solver.problem;
import solver.stage;

// ─────────────────────────────────────────────────────────────────────────────
// The problem
// ─────────────────────────────────────────────────────────────────────────────

/// Two physical parameters, a width in 1 .. 10 and a log-scaled rate in
/// 0.1 .. 100. Maximise a smooth objective; width × rate ≤ 200 is hard,
/// width ≥ 2 terminal, rate ≤ 40 soft at a cost of 0.05 per unit over.
struct Beam
{
    static immutable GeneSpec[2] spec = [GeneSpec("width", 1, 10, 5),
                                         GeneSpec("rate", 0.1, 100, 1, Scale.log)];
    static immutable ConstraintSpec[3] limits = [ConstraintSpec("load", Tier.hard),
                                                 ConstraintSpec("min_width", Tier.terminal),
                                                 ConstraintSpec("rate_cap", Tier.soft, 0.05)];

    /// One evaluation: a plain value type (no references), so copies share nothing.
    struct Eval
    {
        bool   scored;
        double objective = double.nan;
        Breach breach;
        double width, rate;
    }

    /// Why candidates broke their limits, for progress reports.
    struct Causes
    {
        size_t load, minWidth;
        void opOpAssign(string op : "+")(const Causes c) pure nothrow @nogc @safe
        {
            load += c.load;
            minWidth += c.minWidth;
        }
    }

    size_t dimension() const { return spec.length; }

    Eval[] evaluate(const(double[])[] genes, Scoring scoring)
    {
        auto o = new Eval[genes.length];
        foreach (k, g; genes)
        {
            // toPhysical clamps: the searches pass genes as sampled, which
            // may lie outside [0, 1].
            immutable double w = spec[0].toPhysical(g[0]), r = spec[1].toPhysical(g[1]);
            immutable double[3] v = [w * r > 200 ? w * r - 200 : 0, w < 2 ? 2 - w : 0, r > 40 ? r - 40 : 0];
            Eval e = {width: w, rate: r, breach: breachOf(limits[], v[])};
            e.scored = scoring == Scoring.all || e.breach.hard == 0;
            if (e.scored)
                e.objective = 10 - (w - 7) ^^ 2 / 4 - (r - 30) ^^ 2 / 100;
            o[k] = e;
        }
        return o;
    }

    bool   scored(const ref Eval e) const { return e.scored; }
    double objective(const ref Eval e) const { return e.objective; }
    Breach breach(const ref Eval e) const { return e.breach; }

    // Where a candidate lands in the archive: its width and rate.
    size_t descriptorCount() const { return 2; }
    void descriptors(const ref Eval e, double[] d) const { d[0] = e.width; d[1] = e.rate; }

    Causes newCauses() const { return Causes.init; }
    void countCauses(ref Causes c, const ref Eval e) const
    {
        c.load += e.breach.hard > 0;
        c.minWidth += e.breach.terminal > 0;
    }

    string noFeasibleSeed() const { return "start from a narrower beam or a lower rate"; }

    /// What a saved run must match to be resumed on this problem.
    const(string[2])[] fingerprint() const { return [["problem", "beam 1"], ["load_limit", "200"]]; }

    // What the learned model learns of an evaluation: genes in; the
    // objective, the breach over the hard and terminal limits, and the two
    // descriptors out. With these the problem is its own learning spec.
    size_t featureCount() const { return 2; }
    void features(const(double)[] genes, double[] f) const { f[] = genes[]; }
    size_t targetCount() const { return 4; }

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

static assert(isProblem!Beam && hasDescriptors!Beam && hasCauses!Beam && hasSurrogate!Beam && isLearning!Beam);

// The manual shows checkpoint.d whole, as the template to copy; the two
// must not drift apart. (Line ends dropped, for a checkout with CRLF.)
static assert(() {
    import std.array : replace;
    import std.string : indexOf;
    enum string open = "<!-- checkpoint.d -->\n```d\n";
    immutable string manual = import("manual.md").replace("\r", "");
    immutable ptrdiff_t from = manual.indexOf(open);
    if (from < 0)
        return false;
    immutable string shown = manual[from + open.length .. $];
    immutable string file = import("example/source/checkpoint.d").replace("\r", "");
    return shown.length > file.length && shown[0 .. file.length] == file && shown[file.length .. $].indexOf("```") == 0;
}(), "docs/manual.md's copy of checkpoint.d differs from docs/example/source/checkpoint.d");

// ─────────────────────────────────────────────────────────────────────────────
// The run
// ─────────────────────────────────────────────────────────────────────────────

alias Model = LearnedModel!Beam;
alias A = Archiving!(Beam, BinnedGrid);

static immutable Axis[2] axes = [Axis("width", [3, 5, 7], "nmwW", 0, 2),
                                 Axis("rate", [1, 10, 40], "slfF", 0, 10)];

ArchiveOptions archiveOptions()
{
    ArchiveOptions o = {evals: 4000, emitters: 4, explorers: 2, lambda: 8, seed: 9,
                        surrogatePool: 4, surrogateFrom: 3, surrogateEpochs: 5, surrogateAim: true};
    return o;
}

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

void main()
{
    // 1. A CMA-ES stage from the neutral point, pre-screened by a model.
    Beam p;
    Best!(Beam.Eval) best;
    double[] x0 = [Beam.spec[0].toGene(Beam.spec[0].neutral), Beam.spec[1].toGene(Beam.spec[1].neutral)];
    StageOptions so = {lambda: 8, sigma: 0.2, budget: 800, seed: 42, surrogatePool: 4};
    const end = cmaStage(p, best, x0, so, new Model(Beam(), 3, null), (const ref StageGeneration!Beam g) {
        if (g.generation % 25 == 0)
            writefln("stage gen %3d  best %.4f  infeasible %d (load %d, width %d)", g.generation, g.best,
                     g.infeasible, g.causes.load, g.causes.minWidth);
    });
    writefln("stage: %d generations, %d evaluations, stalled %s; best %.4f at width %.3f rate %.3f",
             end.generations, end.spent, end.stalled, valueOf(p, best.eval), best.eval.width, best.eval.rate);

    // 2. The archive over a 4 × 4 grid of width and rate, seeded with the
    //    stage's best: once straight through, once stopped after 10
    //    iterations, saved, loaded and resumed.
    const grid = BinnedGrid(axes[]);
    const ao = archiveOptions();
    const seeds = [best.genes];
    const whole = A.runArchive(p, grid, seeds, ao, null, null, null, null, new Model(Beam(), 5, null));

    immutable path = buildPath(tempDir, "solver-example.state");
    {
        auto model = new Model(Beam(), 5, null);
        const stopped = A.runArchive(p, grid, seeds, ao, null, null, (const ref A.Run r) => r.iterations == 10,
                                     null, model);
        save(path, p, stopped, model);
        writefln("stopped after %d iterations: %d cells, QD score %.2f", stopped.iterations,
                 stopped.archive.coverage, stopped.archive.qdScore);
    }
    auto model = new Model(Beam(), 5, null);
    auto run = load(path, p, model);
    remove(path);
    const resumed = A.runArchive(p, grid, null, ao, null, &run, null, null, model);

    writefln("resumed to %d evaluations: %d of %d cells, best %.4f, %d explorer targets reached",
             resumed.evals, resumed.archive.coverage, grid.cells, resumed.archive.bestScore, resumed.reached.length);
    foreach (c, ref el; whole.archive.elites)
        enforce(el.occupied == resumed.archive.elites[c].occupied && el.genes == resumed.archive.elites[c].genes
                && el.value is resumed.archive.elites[c].value, "the resumed run differs");
    enforce(whole.archive.threshold == resumed.archive.threshold && whole.evals == resumed.evals,
            "the resumed run differs");
    writefln("the resumed run's archive is the uninterrupted run's, bit for bit");
    foreach (c; selectDiverse(resumed.archive, grid, 3, 4))
    {
        const el = &resumed.archive.elites[c];
        writefln("  %s  value %.4f  width %.3f  rate %.3f", grid.code(c), el.value, el.eval.width, el.eval.rate);
    }
}
