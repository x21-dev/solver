/// The problem interface: what a search (a CMA-ES stage, the archive)
/// needs to know about the problem it optimises, and nothing more.
///
/// DRAFT for review (docs/problem-interface.md). Nothing uses it yet.
///
/// A problem is a compile-time concept, not a class: the searches are
/// templates over `P`, so every call into the problem is direct and can be
/// inlined, and the problem keeps its own evaluation type `P.Eval` (pleaser
/// keeps its `Evaluation` as it is, with its own sums in its own order).
/// That is how pleaser runs through it with no loss of speed and the same
/// bits. A problem that wants dynamic dispatch (one written in Python and
/// reached through the C API, or a quick test) derives from `DynamicProblem`
/// below, which is a `P` like any other.
///
/// The search works in normalised gene space: `dimension` genes, each in
/// [0, 1] when evaluated. CMA-ES may sample outside; the search clamps the
/// genes before `evaluate` and adds `Ranking.boundPenalty` times how far
/// out they were (`boundExcess`) to the candidate's fitness, so the problem
/// never sees a gene outside [0, 1]. Mapping genes to physical values (dB,
/// Hz, a quarter, a volume) is the problem's own business; `GeneSpec` is a
/// helper for it.
///
/// Required of `P` (checked by `isProblem`):
///
/// - `alias Eval`: one evaluation. A value type: copying it must not alias
///   anything the problem changes later.
/// - `size_t dimension() const`: the number of searched genes.
/// - `Eval[] evaluate(const(double[])[] genes, Scoring scoring)`: a batch,
///   results in input order. Deterministic: the same genes give the same
///   bits whatever the batch size, the order of the other candidates and
///   the thread count. The problem may run the batch in parallel itself.
/// - `bool scored(const ref Eval) const`: the objective was computed (with
///   `Scoring.feasibleOnly`, a candidate over a hard limit may skip it).
/// - `double objective(const ref Eval) const`: higher is better. Read only
///   when `scored`.
/// - `Breach breach(const ref Eval) const`: how far over its limits, per
///   tier; 0 within them.
///
/// Optional (each checked with `__traits(hasMember)`; a search that needs
/// one says so with a static assert):
///
/// - Archive: `size_t descriptorCount() const` and
///   `void descriptors(const ref Eval, double[] d) const`: where it lands;
///   cells come from a grid (`isGrid`) over these values.
/// - Pre-screening penalty: `double screenPenalty(const(double)[] genes)`:
///   a known cost subtracted from a predicted objective (pleaser's
///   processing cost, which needs no render).
/// - Cause counts for progress reports: `alias Causes` (a value with
///   `opOpAssign!"+"`), `Causes newCauses() const`,
///   `void countCauses(ref Causes, const ref Eval) const`.
/// - Surrogate: `size_t featureCount() const`,
///   `void features(const(double)[] genes, double[] f) const` (default:
///   the genes), `size_t targetCount() const`,
///   `void targets(const ref Eval, double[] t) const` (what the model
///   learns; NaN where not measured), and
///   `Predicted predicted(const(double)[] t) const`: a predicted target
///   row read as objective, breach and descriptors.
/// - Saving: `const(string[2])[] fingerprint() const` (what a saved run
///   must match to be resumed: stems, ranges, limits), and
///   `void saveEval(const ref Eval, ref string[2][] fields) const` with
///   `Eval loadEval(const(string[2])[] fields) const`, for evaluations kept
///   in a checkpoint (an archive's elites) without evaluating them again.
module solver.problem;

import std.math : exp, isNaN, log;
import std.traits : ReturnType;

// ─────────────────────────────────────────────────────────────────────────────
// Constraints and ranking
// ─────────────────────────────────────────────────────────────────────────────

/// The three kinds of limit (the design doc's constraint model).
///
/// - hard: structure and survival; any breach ranks a candidate below
///   every candidate within them (pleaser's limiter and key-track margins
///   are all hard).
/// - terminal: credibility conditions on the outcome (compliance restored
///   by a horizon); ranked after hard, before the objective.
/// - soft: limits that may give at a cost, traded against the objective:
///   `value` is the objective less the soft breach.
enum Tier : ubyte { hard, terminal, soft }

/// How far an evaluation is over its limits, per tier, each already summed
/// (and for soft, weighted) by the problem in the order it chooses: so a
/// problem keeps its own bits. 0 within the limits; never negative.
struct Breach
{
    double hard = 0, terminal = 0, soft = 0;

    /// Within every hard and terminal limit.
    bool feasible() const pure nothrow @nogc @safe { return hard == 0 && terminal == 0; }
}

/// One limit, for reports and for problems that let `breachOf` do the sums.
struct ConstraintSpec
{
    string name;
    Tier   tier;
    double weight = 1;        // soft only: cost per unit of breach
    string group;             // for cause counts, e.g. "key", "floor", "limiter"
}

/// Each tier's breach from per-limit `violations` (one per spec, 0 within),
/// summed in spec order, soft ones weighted. For problems that have no
/// order of their own to keep.
Breach breachOf(const(ConstraintSpec)[] specs, const(double)[] violations) pure nothrow @nogc @safe
{
    assert(specs.length == violations.length);
    Breach b;
    foreach (i, ref s; specs)
    {
        immutable double v = violations[i];
        final switch (s.tier)
        {
            case Tier.hard:     b.hard += v; break;
            case Tier.terminal: b.terminal += v; break;
            case Tier.soft:     b.soft += s.weight * v; break;
        }
    }
    return b;
}

/// What the searches rank by: the objective less the soft breach.
double value(double objective, const Breach b) pure nothrow @nogc @safe
{
    return objective - b.soft;
}

/// How CMA-ES ranks candidates (lower is better): over a hard limit,
/// `hardOffset` plus how far; else over a terminal one, `terminalOffset`
/// plus how far; else minus its value. Plus `boundPenalty` times how far
/// its genes were out of [0, 1].
///
/// The offsets must put every breach above every value: above minus the
/// lowest value, and `hardOffset` above `terminalOffset` plus the largest
/// terminal breach. pleaser's score is 0 .. 28 and it has hard limits only:
/// `Ranking.init` is its `genome.fitness`, bit for bit.
struct Ranking
{
    double hardOffset = 100;      // pleaser's INFEASIBLE
    double terminalOffset = 50;
    double boundPenalty = 10;     // pleaser's BOUND_PENALTY, per squared gene unit

    double fitness(const Breach b, double value, double excess) const pure nothrow @nogc @safe
    {
        return (b.hard > 0 ? hardOffset + b.hard : b.terminal > 0 ? terminalOffset + b.terminal : -value)
             + boundPenalty * excess;
    }
}

/// Whether a batch's candidates over a hard limit must be scored too.
/// `feasibleOnly` lets the problem skip the objective where the ranking
/// doesn't read it (pleaser skips the analyser for mixes over a limit).
enum Scoring : ubyte { all, feasibleOnly }

/// A gene clamped to [0, 1] (NaN stays NaN).
double clamp01(double g) pure nothrow @nogc @safe
{
    return g < 0 ? 0 : (g > 1 ? 1 : g);
}

/// How far genes are out of [0, 1]: the sum of the squared overshoots.
double boundExcess(const(double)[] x) pure nothrow @nogc @safe
{
    double excess = 0;
    foreach (v; x)
    {
        immutable double o = v < 0 ? -v : (v > 1 ? v - 1 : 0);
        excess += o * o;
    }
    return excess;
}

// ─────────────────────────────────────────────────────────────────────────────
// Genes
// ─────────────────────────────────────────────────────────────────────────────

enum Scale : ubyte { linear, log, integer }

/// One gene's physical range, a helper for problems: gene 0 is `lo`, 1 is
/// `hi`. `integer` rounds to the nearest whole value in [lo, hi] (a start
/// quarter, a block count); `log` needs lo > 0.
struct GeneSpec
{
    string name;
    double lo, hi, neutral;
    Scale  scale;

    /// A range of a single value: the gene is left out of the search.
    bool fixed() const pure nothrow @nogc @safe { return lo == hi; }

    double toPhysical(double g) const pure nothrow @nogc @safe
    {
        g = clamp01(g);
        final switch (scale)
        {
            case Scale.linear:  return lo + g * (hi - lo);
            case Scale.log:     return lo * exp(g * log(hi / lo));
            case Scale.integer:
                import std.math : floor;
                return floor(lo + g * (hi - lo) + 0.5);
        }
    }

    /// The gene of value `v`. Not for a fixed range: it has no gene.
    double toGene(double v) const pure nothrow @nogc @safe
    {
        assert(!fixed, "toGene of a fixed gene");
        return clamp01(scale == Scale.log ? log(v / lo) / log(hi / lo) : (v - lo) / (hi - lo));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The concept
// ─────────────────────────────────────────────────────────────────────────────

/// A predicted target row, read as the searches use it (pre-screening,
/// model-guided explorers). `descriptors` may be null when the problem has
/// none.
struct Predicted
{
    double   objective;
    Breach   breach;           // with whatever headroom the problem wants on a prediction
    double[] descriptors;
}

/// Whether `P` is a problem: the required members, with these signatures.
enum bool isProblem(P) = is(P.Eval) && is(typeof((ref P p, const ref P cp, const(double[])[] genes,
                                                    const ref P.Eval e)
{
    size_t n = cp.dimension();
    P.Eval[] batch = p.evaluate(genes, Scoring.all);
    bool s = cp.scored(e);
    double o = cp.objective(e);
    Breach b = cp.breach(e);
}));

/// Whether `P` places evaluations in an archive.
enum bool hasDescriptors(P) = isProblem!P && is(typeof((const ref P cp, const ref P.Eval e, double[] d)
{
    size_t n = cp.descriptorCount();
    cp.descriptors(e, d);
}));

/// Whether `P` can be modelled by a surrogate.
enum bool hasSurrogate(P) = isProblem!P && is(typeof((const ref P cp, const ref P.Eval e,
                                                      const(double)[] x, double[] f)
{
    size_t nf = cp.featureCount(), nt = cp.targetCount();
    cp.features(x, f);
    cp.targets(e, f);
    Predicted q = cp.predicted(x);
}));

/// Whether `P` keeps evaluations and its identity in saved runs.
enum bool isSaveable(P) = isProblem!P && is(typeof((const ref P cp, const ref P.Eval e,
                                                    ref string[2][] fields)
{
    const(string[2])[] fp = cp.fingerprint();
    cp.saveEval(e, fields);
    P.Eval back = cp.loadEval(fields);
}));

/// The value a search ranks `e` by: its objective less its soft breach;
/// NaN when it wasn't scored.
double valueOf(P)(const ref P p, const ref P.Eval e) if (isProblem!P)
{
    return p.scored(e) ? value(p.objective(e), p.breach(e)) : double.nan;
}

/// `e`'s fitness for CMA-ES (lower is better), as the stages and emitters
/// rank it: `genes` as sampled, before clamping.
double fitnessOf(P)(const ref P p, const ref P.Eval e, const(double)[] genes, const ref Ranking r)
    if (isProblem!P)
{
    immutable Breach b = p.breach(e);
    return r.fitness(b, b.feasible ? value(p.objective(e), b) : 0, boundExcess(genes));
}

// ─────────────────────────────────────────────────────────────────────────────
// Archive cells
// ─────────────────────────────────────────────────────────────────────────────

/// Whether `G` is a grid of archive cells over descriptor values:
///
/// - `size_t cells() const`;
/// - `size_t cellOf(const(double)[] d) const`: the cell of descriptors `d`;
/// - `double distance(const(double)[] d, size_t cell) const`: how far `d`
///   is from `cell`, 0 inside, in bin widths (the explorers head down it);
/// - `long binDistance2(size_t a, size_t b) const`: squared distance
///   between cells in bins (the diverse selection, explorer starts);
/// - `string code(size_t cell) const` and `size_t parse(string) const`:
///   the cell's name in files and on the command line (pleaser's "Agn").
///
/// A problem whose cells are not a regular grid (pleaser's channel bins at
/// v * 28 / 7, its tilt with NaN neutral) brings its own `G`; `BinnedGrid`
/// is the default for the rest.
enum bool isGrid(G) = is(typeof((const ref G g, const(double)[] d, string s)
{
    size_t n = g.cells();
    size_t c = g.cellOf(d);
    double x = g.distance(d, c);
    long b = g.binDistance2(c, c);
    string name = g.code(c);
    size_t back = g.parse(s);
}));

/// One axis of a `BinnedGrid`: values below `edges[0]` in bin 0, from
/// `edges[i - 1]` up to `edges[i]` in bin i, from the last edge up in the
/// last bin; NaN in `nanBin`. Each bin's name is one letter, for codes.
struct Axis
{
    string   name;
    double[] edges;           // ascending; edges.length + 1 bins
    string   letters;         // one per bin
    size_t   nanBin;
    double   width = 1;       // the unit of `distance` on this axis

    size_t bins() const pure nothrow @nogc @safe { return edges.length + 1; }

    size_t binOf(double v) const pure nothrow @nogc @safe
    {
        if (isNaN(v))
            return nanBin;
        size_t b = 0;
        while (b < edges.length && v >= edges[b])
            ++b;
        return b;
    }
}

/// A grid of axes, the first axis fastest in the cell index.
struct BinnedGrid
{
    const(Axis)[] axes;

    size_t cells() const pure nothrow @nogc @safe
    {
        size_t n = 1;
        foreach (ref a; axes)
            n *= a.bins;
        return n;
    }

    size_t cellOf(const(double)[] d) const pure nothrow @nogc @safe
    {
        assert(d.length == axes.length);
        size_t c = 0, stride = 1;
        foreach (i, ref a; axes)
        {
            c += a.binOf(d[i]) * stride;
            stride *= a.bins;
        }
        return c;
    }

    /// The bin of `cell` on axis `i`.
    size_t binAt(size_t cell, size_t i) const pure nothrow @nogc @safe
    {
        foreach (j; 0 .. i)
            cell /= axes[j].bins;
        return cell % axes[i].bins;
    }

    double distance(const(double)[] d, size_t cell) const pure nothrow @nogc @safe
    {
        import std.math : sqrt;
        double s = 0;
        foreach (i, ref a; axes)
        {
            immutable size_t want = binAt(cell, i);
            immutable double v = isNaN(d[i]) ? double.nan : d[i];
            if (a.binOf(v) == want)
                continue;
            immutable double lo = want == 0 ? -double.infinity : a.edges[want - 1];
            immutable double hi = want == a.edges.length ? double.infinity : a.edges[want];
            immutable double x = (v < lo ? lo - v : v - hi) / a.width;
            immutable double o = x > 1e-6 ? x : 1e-6;     // on an upper edge: just outside
            s += o * o;
        }
        return sqrt(s);
    }

    long binDistance2(size_t a, size_t b) const pure nothrow @nogc @safe
    {
        long s = 0;
        foreach (i; 0 .. axes.length)
        {
            immutable long d = cast(long) binAt(a, i) - cast(long) binAt(b, i);
            s += d * d;
        }
        return s;
    }

    string code(size_t cell) const pure @safe
    {
        char[] o;
        foreach (i, ref a; axes)
            o ~= a.letters[binAt(cell, i)];
        return o.idup;
    }

    size_t parse(string s) const pure @safe
    {
        import std.exception : enforce;
        import std.string : indexOf;
        enforce(s.length == axes.length, "cell '" ~ s ~ "': one letter per axis");
        size_t c = 0, stride = 1;
        foreach (i, ref a; axes)
        {
            immutable ptrdiff_t b = a.letters.indexOf(s[i]);
            enforce(b >= 0, "cell '" ~ s ~ "': no bin '" ~ s[i] ~ "' on axis " ~ a.name);
            c += b * stride;
            stride *= a.bins;
        }
        return c;
    }
}

static assert(isGrid!BinnedGrid);

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic problems
// ─────────────────────────────────────────────────────────────────────────────

/// A generic evaluation, for problems without a type of their own (the C
/// API, Python, tests).
struct Outcome
{
    bool     scored;
    double   objective = double.nan;
    double[] violations;      // one per ConstraintSpec, 0 within
    Breach   breach;          // breachOf(constraints, violations), or the problem's own sums
    double[] descriptors;     // one per axis
    double[] targets;         // surrogate targets; NaN where not measured
}

/// A problem by virtual calls, for implementations across a language
/// boundary: one call per batch, so the dispatch costs nothing next to an
/// evaluation. Satisfies `isProblem`, `hasDescriptors` and `isSaveable`.
abstract class DynamicProblem
{
    alias Eval = Outcome;

    const(GeneSpec)[]       genes() const;
    const(ConstraintSpec)[] constraints() const;
    const(string[2])[]      fingerprint() const;

    /// Fills `outcomes` (one per row of `genes`, in order).
    void evaluateInto(const(double[])[] genes, Scoring scoring, Outcome[] outcomes);

    size_t descriptorCount() const { return 0; }

    final size_t dimension() const { return genes.length; }

    final Outcome[] evaluate(const(double[])[] g, Scoring scoring)
    {
        auto o = new Outcome[g.length];
        evaluateInto(g, scoring, o);
        return o;
    }

    final bool   scored(const ref Outcome e) const { return e.scored; }
    final double objective(const ref Outcome e) const { return e.objective; }
    final Breach breach(const ref Outcome e) const { return e.breach; }
    final void   descriptors(const ref Outcome e, double[] d) const { d[] = e.descriptors[]; }

    final void saveEval(const ref Outcome e, ref string[2][] fields) const
    {
        import std.format : format;
        fields ~= ["scored", e.scored ? "1" : "0"];
        fields ~= ["objective", format("%a", e.objective)];
        fields ~= ["breach", format("%a %a %a", e.breach.hard, e.breach.terminal, e.breach.soft)];
        fields ~= ["violations", format("%(%a %)", e.violations)];
        fields ~= ["descriptors", format("%(%a %)", e.descriptors)];
    }

    final Outcome loadEval(const(string[2])[] fields) const
    {
        import std.algorithm : map;
        import std.array : array, split;
        import std.conv : to;
        Outcome e;
        foreach (kv; fields)
        {
            auto v = kv[1].split.map!(s => s.to!double).array;
            switch (kv[0])
            {
                case "scored":      e.scored = kv[1] == "1"; break;
                case "objective":   e.objective = v[0]; break;
                case "breach":      e.breach = Breach(v[0], v[1], v[2]); break;
                case "violations":  e.violations = v; break;
                case "descriptors": e.descriptors = v; break;
                default: break;
            }
        }
        return e;
    }
}

static assert(isProblem!DynamicProblem && hasDescriptors!DynamicProblem && isSaveable!DynamicProblem);

/// The first entry of `saved` that `now` lacks or holds differently, as a
/// message; null when they agree. Order doesn't matter; entries only in
/// `now` are new and fine.
string fingerprintMismatch(const(string[2])[] saved, const(string[2])[] now) pure @safe
{
    foreach (s; saved)
    {
        bool found = false;
        foreach (n; now)
            if (n[0] == s[0])
            {
                if (n[1] != s[1])
                    return s[0] ~ ": saved " ~ s[1] ~ ", now " ~ n[1];
                found = true;
                break;
            }
        if (!found)
            return s[0] ~ ": saved " ~ s[1] ~ ", now none";
    }
    return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

version (unittest)
{
    /// pleaser's genome.fitness, copied as it is (pleaser a0fb32f), to hold
    /// Ranking.init to it.
    private double pleaserFitness(double violation, double gain, double excess) pure nothrow @nogc @safe
    {
        return (violation > 0 ? 100 + violation : -gain) + 10 * excess;
    }

    /// A toy problem with its own evaluation type, as pleaser's would be:
    /// maximise -(x - 0.7)² - (y - 0.4)² with x + y ≤ 1 hard, y ≥ 0.2
    /// terminal and x ≤ 0.6 soft (weight 2). The descriptors are x and y.
    private struct Toy
    {
        struct Eval { bool scored; double obj = double.nan; double over, under, soft; double x, y; }

        size_t calls;

        size_t dimension() const { return 2; }

        Eval[] evaluate(const(double[])[] genes, Scoring scoring)
        {
            auto o = new Eval[genes.length];
            foreach (k, g; genes)
            {
                ++calls;
                immutable double x = g[0], y = g[1];
                Eval e;
                e.x = x;
                e.y = y;
                e.over = x + y > 1 ? x + y - 1 : 0;
                e.under = y < 0.2 ? 0.2 - y : 0;
                e.soft = x > 0.6 ? x - 0.6 : 0;
                e.scored = scoring == Scoring.all || e.over == 0;
                if (e.scored)
                    e.obj = -(x - 0.7) ^^ 2 - (y - 0.4) ^^ 2;
                o[k] = e;
            }
            return o;
        }

        bool   scored(const ref Eval e) const { return e.scored; }
        double objective(const ref Eval e) const { return e.obj; }
        Breach breach(const ref Eval e) const { return Breach(e.over, e.under, 2 * e.soft); }

        size_t descriptorCount() const { return 2; }
        void   descriptors(const ref Eval e, double[] d) const { d[0] = e.x; d[1] = e.y; }
    }

    static assert(isProblem!Toy && hasDescriptors!Toy && !hasSurrogate!Toy && !isSaveable!Toy);
}

unittest
{
    // Ranking.init is pleaser's fitness, bit for bit, over hard limits only.
    immutable Ranking r;
    foreach (v; [0.0, 1e-9, 0.5, 3, 250])
        foreach (gain; [-0.0, 0.0, 1e-3, 13.75, 27.9])
            foreach (excess; [0.0, 1e-12, 0.04, 2])
            {
                immutable double ours = r.fitness(Breach(v), value(gain, Breach(v)), excess);
                immutable double theirs = pleaserFitness(v, gain, excess);
                assert((cast(const(ubyte)[])[ours]) == (cast(const(ubyte)[])[theirs]));
            }
}

unittest
{
    // The tiers rank hard, then terminal, then value.
    Toy p;
    auto e = p.evaluate([[0.7, 0.25], [0.9, 0.3], [0.5, 0.1], [0.55, 0.3]], Scoring.feasibleOnly);
    immutable Ranking r;
    double f(size_t k) { return fitnessOf(p, e[k], [e[k].x, e[k].y], r); }
    assert(!e[1].scored && isNaN(valueOf(p, e[1])));       // over the hard limit: not scored
    assert(f(1) > f(2) && f(2) > f(0) && f(2) > f(3));     // hard above terminal above any value
    // The soft limit trades: [0.7, 0.25] has the better objective but pays
    // 2 × 0.1 for x over 0.6; [0.55, 0.3] pays nothing.
    import std.math : isClose;
    assert(p.objective(e[0]) > p.objective(e[3]) && isClose(valueOf(p, e[0]), -0.2225) && f(3) < f(0));
}

unittest
{
    // Evaluations don't depend on the batch: one at a time and all at once.
    Toy p;
    immutable double[][] g = [[0.1, 0.2], [0.3, 0.9], [0.99, 0.01], [0.5, 0.5]];
    auto all = p.evaluate(g, Scoring.all);
    foreach (k, row; g)
        assert(p.evaluate([row], Scoring.all)[0] == all[k]);
}

unittest
{
    // A binned grid: pleaser's tilt axis (warm, neutral, bright, NaN
    // neutral) and a 0 .. 1 channel in quarters, 1 and up last.
    immutable Axis channel = Axis("green", [0.25, 0.5, 0.75, 1.0], "zungA", 0, 0.25);
    immutable Axis tilt = Axis("tilt", [-5.25, -3.75], "wnb", 1, 1.5);
    const g = BinnedGrid([channel, tilt]);
    assert(g.cells == 15);
    assert(g.code(g.cellOf([0.8, double.nan])) == "gn" && g.parse("gn") == g.cellOf([0.8, -4.5]));
    assert(g.cellOf([1.0, -6]) == g.parse("Aw") && g.cellOf([-0.1, -3]) == g.parse("zb"));
    assert(g.distance([0.8, -4.5], g.parse("gn")) == 0);
    assert(g.distance([0.3, -4.5], g.parse("gn")) == (0.75 - 0.3) / 0.25);
    assert(g.binDistance2(g.parse("zw"), g.parse("Ab")) == 16 + 4);
}

unittest
{
    // breachOf sums in spec order, weighting soft limits.
    immutable ConstraintSpec[] specs = [ConstraintSpec("gr_avg", Tier.hard), ConstraintSpec("gr_max", Tier.hard),
                                        ConstraintSpec("by_T", Tier.terminal),
                                        ConstraintSpec("lcr", Tier.soft, 3)];
    immutable Breach b = breachOf(specs, [0.5, 0.25, 1, 0.5]);
    assert(b == Breach(0.75, 1, 1.5) && !b.feasible);
    assert(breachOf(specs, [0, 0, 0, 2]).feasible);
}

unittest
{
    // A dynamic problem saves and loads its evaluations exactly.
    final class Line : DynamicProblem
    {
        override const(GeneSpec)[] genes() const { return [GeneSpec("x", 0, 10, 5)]; }
        override const(ConstraintSpec)[] constraints() const { return [ConstraintSpec("x<=8", Tier.hard)]; }
        override const(string[2])[] fingerprint() const { return [["problem", "line 1"]]; }
        override size_t descriptorCount() const { return 1; }
        override void evaluateInto(const(double[])[] g, Scoring scoring, Outcome[] o)
        {
            foreach (k, row; g)
            {
                immutable double x = genes[0].toPhysical(row[0]);
                o[k].violations = [x > 8 ? x - 8 : 0];
                o[k].breach = breachOf(constraints, o[k].violations);
                o[k].scored = scoring == Scoring.all || o[k].breach.feasible;
                o[k].objective = o[k].scored ? x / 3 : double.nan;
                o[k].descriptors = [x];
            }
        }
    }
    auto p = new Line;
    auto e = p.evaluate([[0.7], [0.9]], Scoring.feasibleOnly);
    assert(e[0].scored && !e[1].scored && e[1].breach.hard > 0);
    string[2][] fields;
    p.saveEval(e[0], fields);
    const back = p.loadEval(fields);
    assert(back.objective == e[0].objective && back.breach == e[0].breach && back.descriptors == e[0].descriptors);
    assert(fingerprintMismatch(p.fingerprint, p.fingerprint) is null);
    assert(fingerprintMismatch([["problem", "line 0"]], p.fingerprint) == "problem: saved line 0, now line 1");
}

unittest
{
    // Genes: an integer gene rounds, a log gene round-trips.
    immutable q = GeneSpec("start_quarter", 1, 8, 1, Scale.integer);
    assert(q.toPhysical(0) == 1 && q.toPhysical(1) == 8 && q.toPhysical(0.5) == 5);
    immutable f = GeneSpec("hpf", 20, 400, 20, Scale.log);
    import std.math : isClose;
    assert(isClose(f.toPhysical(f.toGene(100)), 100));
}
