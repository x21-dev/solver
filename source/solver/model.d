/// A learned model of a problem: an Ensemble trained on every evaluation of
/// a run, which predicts what candidate genes would evaluate to, for the
/// pre-screened stages (StageModel) and the archive's model-guided emitters
/// (ArchiveSurrogate). Moved from pleaser's archivemodel.d, which now gives
/// it only the problem's side: a learning spec.
///
/// A learning spec `L` has
///
/// - `alias Eval`: the evaluation it learns from;
/// - `size_t dimension() const`, `inputs() const`, `outputs() const`,
///   `descriptorCount() const`: genes per candidate, the network's inputs
///   and outputs, the descriptors predicted;
/// - `void features(const(double)[] genes, double[] f) const`: the inputs
///   for genes in [0, 1] (called from several threads at once);
/// - `void targets(const ref Eval e, double[] t) const`: what the network
///   learns of an evaluation, one value per output (NaN where missing;
///   anything not finite counts as missing);
/// - `void predicted(const(double)[] outputs, ref double objective,
///   ref Breach breach, double[] descriptors) const`: the network's outputs
///   as a prediction (also called from several threads at once);
///   `descriptors` is empty when only the objective and breach are asked
///   for (StageModel).
///
/// Deterministic: the features and predictions of a batch are computed one
/// candidate per task, independently, and the ensemble trains as its own
/// documentation says; so a seed gives the same model on any thread count.
module solver.model;

import core.time : MonoTime;
import std.exception : enforce;
import std.format : format;
import std.math : isFinite, sqrt;
import std.parallelism : TaskPool;
import std.range : iota;

import solver.archive : ArchiveSurrogate;
import solver.bytes : Bytes;
import solver.ensemble : Ensemble, EnsembleOptions;
import solver.problem : Breach, clamp01;
import solver.rng : SplitMix64;
import solver.stage : StageModel;

/// Whether `L` is a learning spec (see the module's documentation).
enum bool isLearning(L) = is(L.Eval) && is(typeof((const ref L l, const(double)[] x, double[] f,
                                                    const ref L.Eval e) {
    size_t n = l.dimension() + l.inputs() + l.outputs() + l.descriptorCount();
    l.features(x, f);
    l.targets(e, f);
    double objective;
    Breach b;
    l.predicted(x, objective, b, f);
}));

final class LearnedModel(L) : StageModel!(L.Eval), ArchiveSurrogate!(L.Eval) if (isLearning!L)
{
    alias Eval = L.Eval;
    enum size_t SCALE_ROWS = 4000;     // random gene vectors the inputs are standardised on

    /// A model of `learning`'s problem, its ensemble seeded with `seed`;
    /// `pool` (or null) computes features and predictions and trains.
    this(L learning, ulong seed, TaskPool pool)
    {
        this.learning = learning;
        this.pool = pool;
        EnsembleOptions o;
        o.seed = seed;
        net = new Ensemble(learning.inputs, learning.outputs, o);
        // The inputs' scale, from uniformly random genes: fixed, whatever the data.
        auto rng = SplitMix64(seed ^ 0x5CA1_E5CA_1E5C_A1E5UL);
        auto uni = new double[][](SCALE_ROWS, learning.dimension);
        foreach (ref r; uni)
            foreach (ref v; r)
                v = rng.uniform01();
        const st = columnStats(featuresOf(uni));
        net.scaleInputs(st.mean, st.sd);
    }

    /// Adds evaluated candidates (genes in [0, 1]) to the training data.
    void learn(const(double[])[] genes, const(Eval)[] evals)
    {
        assert(genes.length == evals.length);
        auto f = featuresOf(genes);
        foreach (i, ref e; evals)
        {
            genes_ ~= genes[i].dup;
            x ~= f[i];
            y ~= targetsOf(e);
        }
    }

    /// Trains `epochs` more on all the data so far.
    void train(size_t epochs)
    {
        immutable t0 = MonoTime.currTime;
        net.train(x, y, epochs, pool);
        trainSeconds += (MonoTime.currTime - t0).total!"usecs" / 1e6;
        ++trainings;
    }

    /// StageModel: the predicted objective, and how far over the hard and
    /// terminal limits.
    void predictScore(const(double[])[] genes, double[] score, double[] violation)
    {
        predictEach(genes, (size_t i, const(double)[] m) {
            Breach b;
            learning.predicted(m, score[i], b, null);
            violation[i] = b.hard + b.terminal;
        });
    }

    /// ArchiveSurrogate: the predicted objective, breach and descriptors
    /// (descriptorCount per candidate, flat).
    void predictArchive(const(double[])[] genes, double[] objective, Breach[] breach, double[] descriptors)
    {
        immutable size_t d = learning.descriptorCount;
        predictEach(genes, (size_t i, const(double)[] m) {
            learning.predicted(m, objective[i], breach[i], descriptors[d * i .. d * (i + 1)]);
        });
    }

    /// The network's outputs for candidate genes (clamped to [0, 1] as an
    /// evaluation would take them): `take(i, outputs)` for each, from
    /// several threads at once when there is a pool.
    void predictEach(const(double[])[] genes, scope void delegate(size_t, const(double)[]) take)
    {
        immutable t0 = MonoTime.currTime;
        auto clamped = new double[][](genes.length);
        foreach (i, g; genes)
            clamped[i] = clamp01(g);
        auto f = featuresOf(clamped);
        void one(size_t i)
        {
            auto m = new double[net.outputs];
            net.predict(f[i], m);
            take(i, m);
        }
        if (pool is null)
            foreach (i; 0 .. genes.length)
                one(i);
        else
            foreach (i; pool.parallel(iota(genes.length), 8))
                one(i);
        predictSeconds += (MonoTime.currTime - t0).total!"usecs" / 1e6;
    }

    /// Evaluations learned from.
    size_t rows() const pure nothrow @nogc { return x.length; }

    /// Everything a resumed run needs (the training data and the
    /// ensemble's state), as bytes; `evals`: the run's evaluations so far,
    /// checked when restored.
    ubyte[] save(size_t evals) const pure nothrow
    {
        Bytes o;
        o.put(STATE_VERSION);
        o.put(evals);
        o.put(genes_.length);
        foreach (g; genes_)
            o.put(g);
        foreach (t; y)
            o.put(t);
        o.put(net.save());
        return o.data;
    }

    /// Continues from save()'s bytes, of a run at `evals` evaluations of the
    /// same problem; throws when they don't fit, and then leaves the model
    /// as it was.
    void restore(const(ubyte)[] bytes, size_t evals)
    {
        auto r = Bytes(bytes.dup);
        enforce(r.get!ulong == STATE_VERSION, "the model's state is of another version");
        immutable size_t saved = r.get!size_t;
        enforce(saved == evals, format("the model's state is of evaluation %d, the run's of %d", saved, evals));
        immutable size_t n = r.get!size_t;
        auto g = new double[][](n);
        foreach (ref row; g)
            row = r.getArray!double(learning.dimension);
        auto t = new double[][](n);
        foreach (ref row; t)
            row = r.getArray!double(net.outputs);
        const netBytes = r.getArray!ubyte();
        enforce(r.done, "the model's state has bytes left over");
        net.restore(netBytes);          // all or nothing: on a throw it stays as it was
        genes_ = g;
        x = featuresOf(g);
        y = t;
    }

    /// Wall time spent training and predicting, and how many trainings.
    double trainSeconds = 0, predictSeconds = 0;
    size_t trainings;

private:
    enum ulong STATE_VERSION = 1;
    L learning;
    Ensemble net;
    TaskPool pool;
    double[][] genes_, x, y;

    double[] targetsOf(const ref Eval e)
    {
        auto t = new double[net.outputs];
        learning.targets(e, t);
        foreach (ref v; t)
            v = isFinite(v) ? v : double.nan;
        return t;
    }

    double[][] featuresOf(const(double[])[] genes)
    {
        auto f = new double[][](genes.length, learning.inputs);
        if (pool is null)
            foreach (i; 0 .. genes.length)
                learning.features(genes[i], f[i]);
        else
            foreach (i; pool.parallel(iota(genes.length), 16))
                learning.features(genes[i], f[i]);
        return f;
    }
}

struct ColumnStats
{
    double[] mean, sd;
}

/// Each column's mean and standard deviation (population) over `rows`, in
/// row order.
ColumnStats columnStats(const(double[])[] rows) pure nothrow
{
    assert(rows.length);
    immutable size_t n = rows[0].length;
    auto mean = new double[n], sd = new double[n];
    foreach (i; 0 .. n)
    {
        double s = 0;
        foreach (r; rows)
            s += r[i];
        mean[i] = s / rows.length;
        double v = 0;
        foreach (r; rows)
            v += (r[i] - mean[i]) * (r[i] - mean[i]);
        sd[i] = sqrt(v / rows.length);
    }
    return ColumnStats(mean, sd);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

version (unittest)
{
    /// Two genes, their squares as features; it learns an objective and a
    /// limit's excess, and predicts the genes as descriptors.
    private struct Square
    {
        struct Eval
        {
            double objective, over;
        }
        size_t dimension() const { return 2; }
        size_t inputs() const { return 4; }
        size_t outputs() const { return 2; }
        size_t descriptorCount() const { return 1; }
        void features(const(double)[] x, double[] f) const
        {
            f[0] = x[0];
            f[1] = x[1];
            f[2] = x[0] * x[0];
            f[3] = x[1] * x[1];
        }
        void targets(const ref Eval e, double[] t) const
        {
            t[0] = e.objective;
            t[1] = e.over;
        }
        void predicted(const(double)[] m, ref double objective, ref Breach b, double[] d) const
        {
            objective = m[0];
            b = Breach(m[1] > 0 ? m[1] : 0);
            if (d.length)
                d[0] = m[0] / 2;
        }
    }
    static assert(isLearning!Square);

    private double[][] squareGenes(size_t n, ulong seed)
    {
        auto r = SplitMix64(seed);
        auto g = new double[][](n, 2);
        foreach (ref row; g)
            foreach (ref v; row)
                v = r.uniform01();
        return g;
    }

    private Square.Eval[] squareEvals(const(double[])[] genes)
    {
        auto o = new Square.Eval[genes.length];
        foreach (i, x; genes)
            o[i] = Square.Eval(1 - x[0] * x[0] - x[1] * x[1], x[0] + x[1] - 1);
        return o;
    }
}

unittest
{
    // Column statistics: population sd, in row order.
    const st = columnStats([[1.0, 5], [3.0, 5]]);
    assert(st.mean == [2.0, 5] && st.sd == [1.0, 0]);
}

unittest
{
    // The same calls give the same model with a pool and without, and both
    // predict alike through either interface; missing targets (not finite)
    // count as missing.
    import std.parallelism : TaskPool;
    auto pool = new TaskPool(3);
    scope (exit) pool.finish(true);
    alias M = LearnedModel!Square;
    auto a = new M(Square(), 7, null), b = new M(Square(), 7, pool);
    const g = squareGenes(300, 1);
    auto e = squareEvals(g);
    e[0].objective = double.infinity;
    foreach (m; [a, b])
    {
        m.learn(g, e);
        m.train(3);
    }
    assert(a.save(300) == b.save(300) && a.rows == 300);
    const q = squareGenes(20, 2) ~ [[-0.5, 1.5]];    // the last one clamped
    auto s = new double[q.length], v = new double[q.length];
    auto o = new double[q.length], d = new double[q.length];
    auto br = new Breach[q.length];
    a.predictScore(q, s, v);
    b.predictArchive(q, o, br, d);
    foreach (i; 0 .. q.length)
        assert(s[i] == o[i] && v[i] == br[i].hard && d[i] == o[i] / 2);
    auto c = new double[q.length], w = new double[q.length];
    a.predictScore([[0.0, 1.0]], c[0 .. 1], w[0 .. 1]);
    assert(c[0] == s[$ - 1] && w[0] == v[$ - 1]);
}

unittest
{
    // Saved and restored, a model holds the same rows and network; bytes
    // left over, or of another run, are refused and leave it as it was.
    import std.exception : assertThrown;
    alias M = LearnedModel!Square;
    auto a = new M(Square(), 1, null), b = new M(Square(), 1, null);
    const g = squareGenes(10, 3);
    a.learn(g, squareEvals(g));
    a.train(2);
    b.learn(g[0 .. 1], squareEvals(g[0 .. 1]));
    const bytes = a.save(5), kept = b.save(5);
    assertThrown(b.restore(bytes ~ cast(ubyte) 0, 5));
    assert(b.save(5) == kept);
    assertThrown(b.restore(bytes, 6));
    assert(b.save(5) == kept);
    b.restore(bytes, 5);
    assert(b.save(5) == bytes && b.rows == 10);
}
