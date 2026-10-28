/// A small ensemble of neural networks for surrogate models: what a
/// problem's genes predict of its scores before they are evaluated.
///
/// A few small networks (two ReLU layers), trained by Adam on the
/// evaluations so far and warm-started from one call to the next. Their
/// mean is the prediction; their spread is a rough uncertainty, too narrow
/// to trust as one.
///
/// Deterministic: the members train independently, one task each, every
/// sum in a fixed order, their randomness from solver.rng; so the same
/// calls give the same bits on any thread count. Moved from pleaser's
/// surrogate.d unchanged.
module solver.ensemble;

import std.math : sqrt;
import std.parallelism : TaskPool;

import solver.bytes : Bytes;
import solver.rng : SplitMix64;

struct EnsembleOptions
{
    size_t members = 5;
    size_t hidden = 64;            // units in each of the two hidden layers
    size_t batch = 128;
    double learningRate = 1e-3;    // Adam's
    double l2 = 1e-3;              // weight decay, per minibatch row (as scikit-learn's alpha)
    ulong seed = 1;
}

/// Small multi-output regression networks: inputs → hidden → hidden →
/// outputs, ReLU, trained together on every output. A target may be
/// missing (NaN) in a row, e.g. the score of a mix over a limit: it then
/// adds nothing to the loss. Inputs are standardised with fixed statistics
/// (scaleInputs), targets with the statistics of the rows of the latest
/// train call; when those change, each member's last layer is rescaled so
/// its predictions in the targets' units stay the same, and training goes
/// on from where it was.
final class Ensemble
{
    immutable size_t inputs, outputs;
    immutable EnsembleOptions options;

    this(size_t inputs, size_t outputs, EnsembleOptions o = EnsembleOptions.init) pure nothrow
    {
        assert(inputs > 0 && outputs > 0 && o.members > 0 && o.hidden > 0 && o.batch > 0);
        this.inputs = inputs;
        this.outputs = outputs;
        options = o;
        inMean = new double[inputs];
        inMean[] = 0;
        inScale = new double[inputs];
        inScale[] = 1;
        outMean = new double[outputs];
        outMean[] = 0;
        outSd = new double[outputs];
        outSd[] = 1;
        auto root = SplitMix64(o.seed);
        members = new Member[o.members];
        foreach (ref m; members)
            m.initialize(inputs, o.hidden, outputs, root.next());
    }

    /// The inputs' standardisation: (x - mean) / sd. Fixed for the
    /// ensemble's life: the members learn on it.
    void scaleInputs(const(double)[] mean, const(double)[] sd) pure nothrow @nogc
    {
        assert(mean.length == inputs && sd.length == inputs);
        foreach (i; 0 .. inputs)
        {
            inMean[i] = mean[i];
            inScale[i] = sd[i] > 1e-12 ? 1 / sd[i] : 1;
        }
    }

    /// Trains every member `epochs` passes over the rows, in a fresh random
    /// order each, from where it is. `y[r][o]`: NaN where missing. The
    /// members train in parallel on `pool` when given.
    void train(const(double[])[] x, const(double[])[] y, size_t epochs, TaskPool pool = null)
    {
        import std.range : iota;
        assert(x.length == y.length);
        immutable size_t rows = x.length;
        if (rows == 0 || epochs == 0)
            return;
        updateTargetStats(y);
        const Data d = prepare(x, y);
        if (pool is null)
            foreach (ref m; members)
                m.epochs(d, epochs, options);
        else
            foreach (k; pool.parallel(iota(members.length), 1))
                members[k].epochs(d, epochs, options);
        epochsDone += epochs;
    }

    /// The members' mean prediction for `x`, in the targets' units, and
    /// their spread (standard deviation) when `spread` is given.
    void predict(const(double)[] x, double[] mean, double[] spread = null) const pure nothrow
    {
        assert(x.length == inputs && mean.length == outputs);
        assert(spread.length == 0 || spread.length == outputs);
        auto z = new double[inputs];
        standardise(x, z);
        auto all = new double[members.length * outputs];
        auto h1 = new double[options.hidden], h2 = new double[options.hidden];
        foreach (k, ref m; members)
        {
            auto o = all[k * outputs .. (k + 1) * outputs];
            m.forward(z, h1, h2, o);
            foreach (j; 0 .. outputs)
                o[j] = o[j] * outSd[j] + outMean[j];
        }
        immutable double inv = 1.0 / members.length;
        foreach (j; 0 .. outputs)
        {
            double s = 0;
            foreach (k; 0 .. members.length)
                s += all[k * outputs + j];
            mean[j] = s * inv;
        }
        if (spread.length)
            foreach (j; 0 .. outputs)
            {
                double s = 0;
                foreach (k; 0 .. members.length)
                {
                    immutable double e = all[k * outputs + j] - mean[j];
                    s += e * e;
                }
                spread[j] = sqrt(s * inv);
            }
    }

    /// Epochs trained so far (each member).
    size_t epochsTrained() const pure nothrow @nogc { return epochsDone; }

    /// Everything that changes as it trains, as bytes, for a checkpoint:
    /// an ensemble of the same shape and options restored from them goes on
    /// bit-identically.
    ubyte[] save() const pure nothrow
    {
        Bytes o;
        o.put(inputs);
        o.put(outputs);
        o.put(members.length);
        o.put(options.hidden);
        o.put(haveStats);
        o.put(epochsDone);
        o.put(inMean);
        o.put(inScale);
        o.put(outMean);
        o.put(outSd);
        foreach (ref m; members)
            m.state.write(o);
        return o.data;
    }

    /// Continues from save()'s bytes; throws when they are of another shape,
    /// and then leaves the ensemble as it was.
    void restore(const(ubyte)[] bytes) pure
    {
        import std.exception : enforce;
        auto r = Bytes(bytes.dup);
        immutable size_t ni = r.get!size_t, no = r.get!size_t, nm = r.get!size_t, nh = r.get!size_t;
        enforce(ni == inputs && no == outputs && nm == members.length && nh == options.hidden,
                "the saved ensemble is of another shape");
        // Everything is read into new values first and taken over only once
        // the bytes have all been read: the copies of the members get new
        // arrays, never write into the ones they share with the originals.
        immutable bool stats = r.get!bool;
        immutable size_t done = r.get!size_t;
        auto im = r.getArray!double(inputs), isc = r.getArray!double(inputs);
        auto om = r.getArray!double(outputs), osd = r.getArray!double(outputs);
        auto ms = members.dup;
        foreach (ref m; ms)
            m.state = MemberState.read(r, m.p.length);
        enforce(r.done, "the saved ensemble has bytes left over");
        haveStats = stats;
        epochsDone = done;
        inMean = im;
        inScale = isc;
        outMean = om;
        outSd = osd;
        members = ms;
    }

private:
    Member[] members;
    double[] inMean, inScale, outMean, outSd;
    bool haveStats;
    size_t epochsDone;

    void standardise(const(double)[] x, double[] z) const pure nothrow @nogc
    {
        foreach (i; 0 .. inputs)
            z[i] = (x[i] - inMean[i]) * inScale[i];
    }

    /// Takes the targets' statistics from `y`; the members' last layers
    /// follow when they change.
    void updateTargetStats(const(double[])[] y) pure nothrow
    {
        const st = targetStats(y, outMean, outSd);
        if (haveStats)
            foreach (ref m; members)
                m.rescaleOutputs(outMean, outSd, st.mean, st.sd);
        outMean[] = st.mean[];
        outSd[] = st.sd[];
        haveStats = true;
    }

    /// The rows standardised with the current statistics, and which targets
    /// are present (a missing one's standardised value is 0).
    Data prepare(const(double[])[] x, const(double[])[] y) const pure nothrow
    {
        immutable size_t rows = x.length;
        auto z = new double[rows * inputs];
        auto t = new double[rows * outputs];
        auto present = new bool[rows * outputs];
        foreach (r; 0 .. rows)
        {
            assert(x[r].length == inputs && y[r].length == outputs);
            standardise(x[r], z[r * inputs .. (r + 1) * inputs]);
            foreach (o; 0 .. outputs)
            {
                immutable double v = y[r][o];
                immutable bool p = v == v;
                present[r * outputs + o] = p;
                t[r * outputs + o] = p ? (v - outMean[o]) / outSd[o] : 0;
            }
        }
        return Data(z, t, present, rows);
    }
}

private:

struct Stats
{
    double[] mean, sd;
}

/// Each target's mean and standard deviation (at least 1e-9) over the rows
/// where it is present (not NaN), in row order; `mean0` and `sd0` where it
/// is present in none.
Stats targetStats(const(double[])[] y, const(double)[] mean0, const(double)[] sd0) pure nothrow
{
    assert(mean0.length == sd0.length);
    immutable size_t outputs = mean0.length;
    auto mean = new double[outputs], sd = new double[outputs];
    foreach (o; 0 .. outputs)
    {
        double s = 0;
        size_t c = 0;
        foreach (r; y)
            if (r[o] == r[o])
            {
                s += r[o];
                ++c;
            }
        if (c == 0)
        {
            mean[o] = mean0[o];
            sd[o] = sd0[o];
            continue;
        }
        mean[o] = s / c;
        double v = 0;
        foreach (r; y)
            if (r[o] == r[o])
                v += (r[o] - mean[o]) * (r[o] - mean[o]);
        immutable double d = sqrt(v / c);
        sd[o] = d > 1e-9 ? d : 1e-9;
    }
    return Stats(mean, sd);
}

/// c[m * ld + k] += Σ_s a[m * am + s * as] · b[s * ld + k], for m < mn,
/// k < kn, s from 0 up to sn: each element of c adds its terms in order of
/// s, each as c + a·b, so the result is the same bits as the plain loops.
/// Tiles of MR rows by KR columns are summed in registers, loaded and
/// stored once; what is left of the rows and columns takes plain loops.
void multiplyAdd(double[] c, const(double)[] a, size_t am, size_t as, const(double)[] b,
                 size_t mn, size_t sn, size_t kn, size_t ld) pure nothrow @nogc @trusted
{
    import ldc.simd : loadUnaligned, storeUnaligned;
    enum size_t MR = 4, KV = 2, KR = 4 * KV;
    // One element, plainly.
    void plain(size_t m, size_t k)
    {
        double v = c[m * ld + k];
        foreach (s_; 0 .. sn)
            v = v + a[m * am + s_ * as] * b[s_ * ld + k];
        c[m * ld + k] = v;
    }
    size_t m = 0;
    for (; m + MR <= mn; m += MR)
    {
        size_t k = 0;
        for (; k + KR <= kn; k += KR)
        {
            V[KV][MR] acc;
            static foreach (r; 0 .. MR)
                static foreach (v; 0 .. KV)
                    acc[r][v] = loadUnaligned!V(&c[(m + r) * ld + k + 4 * v]);
            foreach (s_; 0 .. sn)
            {
                V[KV] w;
                static foreach (v; 0 .. KV)
                    w[v] = loadUnaligned!V(&b[s_ * ld + k + 4 * v]);
                static foreach (r; 0 .. MR)
                {{
                    const V x = a[(m + r) * am + s_ * as];
                    static foreach (v; 0 .. KV)
                        acc[r][v] = acc[r][v] + x * w[v];
                }}
            }
            static foreach (r; 0 .. MR)
                static foreach (v; 0 .. KV)
                    storeUnaligned!V(acc[r][v], &c[(m + r) * ld + k + 4 * v]);
        }
        for (; k < kn; ++k)
            foreach (r; 0 .. MR)
                plain(m + r, k);
    }
    for (; m < mn; ++m)
        foreach (k; 0 .. kn)
            plain(m, k);
}

/// One train call's rows: standardised inputs and targets, row-major, and
/// which targets are present.
struct Data
{
    const(double)[] z, t;
    const(bool)[] present;
    size_t rows;
}

/// What a member carries from one train call to the next: all that
/// Ensemble.save keeps of it, in the order it writes it. That order is the
/// checkpoints' format: a field added here changes it.
struct MemberState
{
    SplitMix64 rng;
    double beta1Power = 1, beta2Power = 1;
    // All parameters in one array, so Adam runs over it in one loop:
    // w0[i * nh + k], b0[k], w1[k * nh + m], b1[m], w2[m * nout + o], b2[o];
    // and Adam's moments of each.
    double[] p, am, av;
    size_t[] order;

    void write(ref Bytes o) const pure nothrow
    {
        foreach (v; this.tupleof)
        {
            static if (is(typeof(v) : const SplitMix64))
                o.put(v.state);
            else
                o.put(v);
        }
    }

    /// Reads what write wrote; every double[] is of `params` values.
    static MemberState read(ref Bytes r, size_t params) pure
    {
        MemberState s;
        foreach (ref v; s.tupleof)
        {
            alias T = typeof(v);
            static if (is(T == SplitMix64))
                v = SplitMix64(r.get!ulong);
            else static if (is(T == double[]))
                v = r.getArray!double(params);
            else static if (is(T == size_t[]))
                v = r.getArray!size_t();
            else static if (is(T == double))
                v = r.get!double;
            else
                static assert(0, "MemberState.read: a field of type " ~ T.stringof);
        }
        return s;
    }
}

struct Member
{
    MemberState state;
    alias state this;
    size_t nin, nh, nout;
    double[] g;                    // the gradient, as p
    size_t at0, atb0, at1, atb1, at2, atb2;
    // Scratch for one minibatch.
    double[] h1, h2, y3, d1, d2, d3, sc, zs;

    void initialize(size_t nin, size_t nh, size_t nout, ulong seed) pure nothrow
    {
        this.nin = nin;
        this.nh = nh;
        this.nout = nout;
        at0 = 0;
        atb0 = at0 + nin * nh;
        at1 = atb0 + nh;
        atb1 = at1 + nh * nh;
        at2 = atb1 + nh;
        atb2 = at2 + nh * nout;
        immutable size_t total = atb2 + nout;
        p = new double[total];
        g = new double[total];
        am = new double[total];
        av = new double[total];
        am[] = 0;
        av[] = 0;
        rng = SplitMix64(seed);
        // Glorot-uniform weights and biases, as scikit-learn's MLP.
        void fill(double[] a, size_t fanIn, size_t fanOut)
        {
            immutable double lim = sqrt(6.0 / (fanIn + fanOut));
            foreach (ref v; a)
                v = lim * rng.bipolar();
        }
        fill(p[at0 .. atb0], nin, nh);
        fill(p[atb0 .. at1], nin, nh);
        fill(p[at1 .. atb1], nh, nh);
        fill(p[atb1 .. at2], nh, nh);
        fill(p[at2 .. atb2], nh, nout);
        fill(p[atb2 .. total], nh, nout);
    }

    /// Keeps the outputs in the targets' units when their standardisation
    /// changes from (mean0, sd0) to (mean1, sd1).
    void rescaleOutputs(const(double)[] mean0, const(double)[] sd0, const(double)[] mean1, const(double)[] sd1)
        pure nothrow @nogc
    {
        foreach (o; 0 .. nout)
        {
            immutable double r = sd0[o] / sd1[o];
            foreach (m; 0 .. nh)
                p[at2 + m * nout + o] *= r;
            p[atb2 + o] = p[atb2 + o] * r + (mean0[o] - mean1[o]) / sd1[o];
        }
    }

    /// One row through the network: hidden activations and outputs
    /// (standardised).
    void forward(const(double)[] z, double[] a1, double[] a2, double[] o) const pure nothrow @nogc
    {
        a1[] = p[atb0 .. at1];
        foreach (i; 0 .. nin)
            a1[] += z[i] * p[at0 + i * nh .. at0 + (i + 1) * nh];
        upper(a1, a2, o);
    }

    /// The rest of forward(), from the first layer's sums `a1` (the ReLU
    /// not yet applied, here in place) on.
    void upper(double[] a1, double[] a2, double[] o) const pure nothrow @nogc
    {
        foreach (ref v; a1)
            v = v > 0 ? v : 0;
        // Not multiplyAdd's row tiles, as the first layer: about half of
        // a1 is 0 after the ReLU (a different half in every row), which the
        // loop below skips and the tiles couldn't. They would do twice the
        // terms, which their speed-up per term (about 1.7x) doesn't make up.
        a2[] = p[atb1 .. at2];
        foreach (k; 0 .. nh)
            if (a1[k] != 0)
                a2[] += a1[k] * p[at1 + k * nh .. at1 + (k + 1) * nh];
        foreach (ref v; a2)
            v = v > 0 ? v : 0;
        o[] = p[atb2 .. $];
        foreach (m; 0 .. nh)
            if (a2[m] != 0)
                o[] += a2[m] * p[at2 + m * nout .. at2 + (m + 1) * nout];
    }

    /// Scratch for minibatches of up to `bmax` rows.
    void scratch(size_t bmax) pure nothrow
    {
        if (h1.length == bmax * nh)
            return;
        h1 = new double[bmax * nh];
        h2 = new double[bmax * nh];
        d1 = new double[bmax * nh];
        d2 = new double[bmax * nh];
        y3 = new double[bmax * nout];
        d3 = new double[bmax * nout];
        zs = new double[bmax * nin];
        sc = new double[nout];
    }

    void epochs(const ref Data d, size_t count, const ref EnsembleOptions opt) pure nothrow
    {
        immutable size_t bmax = opt.batch;
        scratch(bmax);
        if (order.length != d.rows)
        {
            order = new size_t[d.rows];
            foreach (r, ref v; order)
                v = r;
        }
        foreach (e; 0 .. count)
        {
            // Fisher-Yates, from this member's own generator.
            for (size_t i = d.rows - 1; i > 0; --i)
            {
                immutable size_t j = cast(size_t)(rng.uniform01() * (i + 1));
                immutable size_t s = order[i];
                order[i] = order[j];
                order[j] = s;
            }
            for (size_t s = 0; s < d.rows; s += bmax)
            {
                gradient(d, order[s .. s + bmax < d.rows ? s + bmax : d.rows], opt.l2);
                adam(opt.learningRate);
            }
        }
    }

    /// The loss's gradient at p on the minibatch `rows`, into g. The loss:
    /// half the squared error, averaged over the present values of each
    /// output, then over the outputs; plus l2 / 2B times the squared
    /// weights (not the biases). Changes only g and the scratch.
    void gradient(const ref Data d, const(size_t)[] rows, double l2) pure nothrow @nogc
    {
        immutable size_t B = rows.length;
        assert(B <= h1.length / nh);
        // Forward, the first layer for the whole minibatch at once (its
        // inputs gathered into zs): a matrix product by multiplyAdd, each
        // row's sums still adding up input by input as forward() adds them.
        foreach (b, r; rows)
        {
            zs[b * nin .. (b + 1) * nin] = d.z[r * nin .. (r + 1) * nin];
            h1[b * nh .. (b + 1) * nh] = p[atb0 .. at1];
        }
        multiplyAdd(h1, zs, nin, 1, p[at0 .. atb0], B, nin, nh, nh);
        foreach (b; 0 .. B)
            upper(h1[b * nh .. (b + 1) * nh], h2[b * nh .. (b + 1) * nh], y3[b * nout .. (b + 1) * nout]);

        sc[] = 0;
        foreach (r; rows)
            foreach (o; 0 .. nout)
                sc[o] += d.present[r * nout + o] ? 1 : 0;
        foreach (o; 0 .. nout)
            sc[o] = 1.0 / ((sc[o] > 0 ? sc[o] : 1) * nout);

        g[] = 0;
        auto gw0 = g[at0 .. atb0], gb0 = g[atb0 .. at1], gw1 = g[at1 .. atb1], gb1 = g[atb1 .. at2];
        auto gw2 = g[at2 .. atb2], gb2 = g[atb2 .. $];
        foreach (b, r; rows)
        {
            const a1 = h1[b * nh .. (b + 1) * nh], a2 = h2[b * nh .. (b + 1) * nh];
            auto e3 = d3[b * nout .. (b + 1) * nout], e2 = d2[b * nh .. (b + 1) * nh], e1 = d1[b * nh .. (b + 1) * nh];
            foreach (o; 0 .. nout)
                e3[o] = d.present[r * nout + o] ? (y3[b * nout + o] - d.t[r * nout + o]) * sc[o] : 0;
            gb2[] += e3[];
            foreach (m; 0 .. nh)
                if (a2[m] != 0)
                    gw2[m * nout .. (m + 1) * nout] += a2[m] * e3[];
            foreach (m; 0 .. nh)
            {
                double s = 0;
                if (a2[m] > 0)
                    foreach (o; 0 .. nout)
                        s += e3[o] * p[at2 + m * nout + o];
                e2[m] = s;
            }
            gb1[] += e2[];
            foreach (k; 0 .. nh)
                if (a1[k] != 0)
                    gw1[k * nh .. (k + 1) * nh] += a1[k] * e2[];
            foreach (k; 0 .. nh)
                e1[k] = a1[k] > 0 ? dot(p[at1 + k * nh .. at1 + (k + 1) * nh], e2) : 0;
            gb0[] += e1[];
        }
        // The first layer's weights: gw0[i][k] += Σ_b z_b[i] e1_b[k], the
        // rows in order, as a matrix product (the inputs transposed).
        multiplyAdd(gw0, zs, 1, nin, d1, nin, B, nh, nh);
        immutable double decay = l2 / B;
        g[at0 .. atb0] += decay * p[at0 .. atb0];
        g[at1 .. atb1] += decay * p[at1 .. atb1];
        g[at2 .. atb2] += decay * p[at2 .. atb2];
    }

    /// One Adam step from the gradient in g: p, the moments and the bias
    /// corrections' powers move on.
    void adam(double lr) pure nothrow @nogc
    {
        beta1Power *= BETA1;
        beta2Power *= BETA2;
        immutable double c1 = 1 / (1 - beta1Power), c2 = 1 / (1 - beta2Power);
        // Locals, not the fields: a store to p could alias this's slices as
        // far as LLVM knows, so it would reload their pointers and the loop
        // would stay scalar.
        auto p = this.p, am = this.am[0 .. p.length], av = this.av[0 .. p.length];
        const g = this.g[0 .. p.length];
        foreach (k; 0 .. p.length)
        {
            immutable double m = BETA1 * am[k] + (1 - BETA1) * g[k], v = BETA2 * av[k] + (1 - BETA2) * (g[k] * g[k]);
            am[k] = m > TINY || m < -TINY ? m : 0;
            av[k] = v > TINY ? v : 0;
            p[k] -= lr * (am[k] * c1) / (sqrt(av[k] * c2) + EPSILON);
        }
    }

    enum double BETA1 = 0.9, BETA2 = 0.999, EPSILON = 1e-8;
    /// Adam's moments below this are set to 0. A parameter whose gradient
    /// stays 0 (the bias of a unit that never fires) would otherwise decay
    /// into subnormal numbers, which cost the CPU many times a normal
    /// operation. Its update is far below EPSILON either way.
    enum double TINY = 1e-150;
}

/// Four doubles, one AVX register.
alias V = __vector(double[4]);

/// Dot product in four independent lanes, combined in a fixed order (as
/// pleaser.audibility's).
double dot(const(double)[] a, const(double)[] b) pure nothrow @nogc
{
    assert(a.length == b.length);
    enum lanes = 4;
    double[lanes] acc = 0;
    size_t i = 0;
    for (; i + lanes <= a.length; i += lanes)
        static foreach (l; 0 .. lanes)
            acc[l] += a[i + l] * b[i + l];
    double tail = 0;
    for (; i < a.length; ++i)
        tail += a[i] * b[i];
    return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + tail;
}

unittest
{
    // The ensemble learns a linear and a product target, and one present in
    // only half the rows; then predicts rows it hasn't seen.
    import std.math : abs;
    auto rng = SplitMix64(7);
    double[][] gen(size_t rows, out double[][] y)
    {
        double[][] x;
        foreach (r; 0 .. rows)
        {
            auto v = [rng.uniform01(), rng.uniform01(), rng.uniform01(), rng.uniform01()];
            x ~= v;
            y ~= [v[0] + 2 * v[1], 4 * v[2] * v[3], r % 2 ? 3 * v[0] - v[3] : double.nan];
        }
        return x;
    }
    double[][] ytr, yte;
    auto xtr = gen(2000, ytr), xte = gen(500, yte);
    EnsembleOptions o;
    o.members = 3;
    o.hidden = 32;
    auto e = new Ensemble(4, 3, o);
    e.scaleInputs([0.5, 0.5, 0.5, 0.5], [0.29, 0.29, 0.29, 0.29]);
    e.train(xtr, ytr, 60);
    assert(e.epochsTrained == 60);
    auto mean = new double[3], spread = new double[3];
    foreach (k; 0 .. 3)
    {
        double se = 0, sv = 0, mu = 0;
        size_t c = 0;
        foreach (r; 0 .. xte.length)
            if (r % 2 || k < 2)
            {
                mu += yte[r][k];
                ++c;
            }
        mu /= c;
        foreach (r; 0 .. xte.length)
            if (r % 2 || k < 2)
            {
                e.predict(xte[r], mean, spread);
                se += (mean[k] - yte[r][k]) ^^ 2;
                sv += (yte[r][k] - mu) ^^ 2;
                assert(spread[k] >= 0);
            }
        assert(1 - se / sv > 0.95);
    }

    // When the targets' statistics change, the predictions in the targets'
    // units stay put (up to rounding).
    auto before = new double[3], after = new double[3];
    e.predict(xte[0], before);
    double[][] shifted;
    foreach (r; ytr)
        shifted ~= [3 * r[0] + 10, r[1] - 5, r[2] * 0.5];
    e.updateTargetStats(shifted);
    e.predict(xte[0], after);
    foreach (k; 0 .. 3)
        assert(abs(after[k] - before[k]) < 1e-9 * (1 + abs(before[k])));
}

unittest
{
    // Bit-identical on any thread count: the members train one task each.
    import std.parallelism : TaskPool;
    auto rng = SplitMix64(3);
    double[][] x, y;
    foreach (r; 0 .. 300)
    {
        x ~= [rng.uniform01(), rng.uniform01()];
        y ~= [x[$ - 1][0] * x[$ - 1][1], r % 3 ? x[$ - 1][0] : double.nan];
    }
    EnsembleOptions o;
    o.members = 4;
    o.hidden = 16;
    o.batch = 32;
    auto a = new Ensemble(2, 2, o), b = new Ensemble(2, 2, o);
    auto pool = new TaskPool(3);
    scope (exit) pool.finish(true);
    foreach (k; 0 .. 3)            // warm-started calls on growing data
    {
        a.train(x[0 .. 100 * (k + 1)], y[0 .. 100 * (k + 1)], 4);
        b.train(x[0 .. 100 * (k + 1)], y[0 .. 100 * (k + 1)], 4, pool);
    }
    auto pa = new double[2], pb = new double[2], sa = new double[2], sb = new double[2];
    foreach (r; 0 .. 20)
    {
        a.predict(x[r], pa, sa);
        b.predict(x[r], pb, sb);
        assert(pa == pb && sa == sb);
    }
}

unittest
{
    // The checkpoints' format: a small ensemble's saved bytes, pinned from
    // before MemberState. If this fails, saved states no longer load.
    {
        import std.digest.sha : sha256Of, toHexString;
        auto rng = SplitMix64(23);
        double[][] x, y;
        foreach (r; 0 .. 40)
        {
            x ~= [rng.uniform01(), rng.uniform01()];
            y ~= [x[$ - 1][0] - x[$ - 1][1], r % 3 ? x[$ - 1][1] : double.nan];
        }
        EnsembleOptions o;
        o.members = 2;
        o.hidden = 3;
        o.batch = 16;
        auto e = new Ensemble(2, 2, o);
        e.scaleInputs([0.5, 0.5], [0.3, 0.3]);
        e.train(x, y, 2);
        const b = e.save();
        assert(b.length == 2281);
        assert(toHexString(sha256Of(b)) == "D2447DFD1DF95CC4F237113DDAC72BE97169F0C2E7F910A6ACEA330F7B56D8F8");
    }

    // Saved after some training and restored into a fresh ensemble, it goes
    // on as the one that never stopped, bit for bit.
    auto rng = SplitMix64(11);
    double[][] x, y;
    foreach (r; 0 .. 200)
    {
        x ~= [rng.uniform01(), rng.uniform01(), rng.uniform01()];
        y ~= [x[$ - 1][0] + x[$ - 1][1] * x[$ - 1][2], r % 4 ? x[$ - 1][2] : double.nan];
    }
    EnsembleOptions o;
    o.members = 2;
    o.hidden = 8;
    o.batch = 16;
    auto a = new Ensemble(3, 2, o);
    a.scaleInputs([0.5, 0.5, 0.5], [0.3, 0.3, 0.3]);
    a.train(x[0 .. 100], y[0 .. 100], 3);
    auto b = new Ensemble(3, 2, o);
    b.restore(a.save());
    assert(b.save() == a.save());
    a.train(x, y, 3);
    b.train(x, y, 3);
    auto pa = new double[2], pb = new double[2];
    foreach (r; 0 .. 10)
    {
        a.predict(x[r], pa);
        b.predict(x[r], pb);
        assert(pa == pb);
    }
    import std.exception : assertThrown;
    assertThrown(new Ensemble(3, 3, o).restore(a.save()));

    // Bytes cut short in the last member, or with one left over: it throws
    // and the ensemble stays as it was.
    auto c = new Ensemble(3, 2, o);
    c.train(x[0 .. 50], y[0 .. 50], 2);
    const kept = c.save();
    assertThrown(c.restore(a.save()[0 .. $ - 8]));
    assert(c.save() == kept);
    assertThrown(c.restore(a.save() ~ cast(ubyte) 0));
    assert(c.save() == kept);
}

unittest
{
    // The targets' statistics: over the present values only, the fallback
    // where none is, and a floor on the spread.
    immutable double nan = double.nan;
    const st = targetStats([[1.0, nan, 4], [3.0, nan, 4], [nan, nan, 4]], [7.0, 8, 0], [9.0, 10, 1]);
    assert(st.mean == [2.0, 8, 4] && st.sd == [1.0, 10, 1e-9]);
}

unittest
{
    // The gradient against central differences of the loss it documents,
    // on a small net with some targets missing; it leaves the parameters
    // and Adam's state alone. Then Adam's first step moves each parameter
    // by the learning rate times the sign of its gradient (nearly).
    import std.math : abs;
    enum size_t nin = 3, nh = 5, nout = 2, rows = 12;
    immutable double l2 = 0.3;
    auto rng = SplitMix64(17);
    auto z = new double[rows * nin], t = new double[rows * nout];
    auto present = new bool[rows * nout];
    foreach (ref v; z)
        v = 2 * rng.bipolar();
    foreach (k, ref v; t)
    {
        present[k] = k % 5 != 3;
        v = present[k] ? rng.bipolar() : 0;
    }
    const d = Data(z, t, present, rows);
    size_t[rows] idx;
    foreach (r, ref v; idx)
        v = r;
    const(size_t)[] batch = idx[];

    Member m;
    m.initialize(nin, nh, nout, 99);
    m.scratch(rows);

    double loss()
    {
        double[nh] a1, a2;
        double[nout] o;
        double[nout] count = 0;
        foreach (r; batch)
            foreach (j; 0 .. nout)
                count[j] += present[r * nout + j] ? 1 : 0;
        double sum = 0;
        foreach (r; batch)
        {
            m.forward(z[r * nin .. (r + 1) * nin], a1[], a2[], o[]);
            foreach (j; 0 .. nout)
                if (present[r * nout + j])
                    sum += 0.5 * (o[j] - t[r * nout + j]) ^^ 2 / ((count[j] > 0 ? count[j] : 1) * nout);
        }
        double w = 0;
        foreach (span; [[m.at0, m.atb0], [m.at1, m.atb1], [m.at2, m.atb2]])
            foreach (v; m.p[span[0] .. span[1]])
                w += v * v;
        return sum + 0.5 * l2 / batch.length * w;
    }

    const p0 = m.p.dup, am0 = m.am.dup, av0 = m.av.dup;
    m.gradient(d, batch, l2);
    assert(m.p == p0 && m.am == am0 && m.av == av0 && m.beta1Power == 1 && m.beta2Power == 1);
    const g = m.g.dup;
    immutable double h = 1e-6;
    foreach (k; 0 .. m.p.length)
    {
        m.p[k] = p0[k] + h;
        immutable double up = loss();
        m.p[k] = p0[k] - h;
        immutable double down = loss();
        m.p[k] = p0[k];
        assert(abs((up - down) / (2 * h) - g[k]) < 1e-7, "gradient off at parameter");
    }

    immutable double lr = 1e-3;
    m.adam(lr);
    assert(m.beta1Power == 0.9 && m.beta2Power == 0.999);
    foreach (k; 0 .. m.p.length)
        assert(abs(p0[k] - m.p[k] - lr * g[k] / (abs(g[k]) + 1e-8)) < 1e-12);
}

unittest
{
    // multiplyAdd: the same bits as the plain loops, tiles and what is left
    // of the rows and columns alike, in both layouts gradient() uses.
    auto rng = SplitMix64(31);
    foreach (dims; [[9, 7, 19], [4, 3, 8], [13, 5, 16], [1, 1, 1]])
    {
        immutable size_t mn = dims[0], sn = dims[1], kn = dims[2], ld = kn + 3;
        auto a = new double[mn * sn], b = new double[sn * ld], c = new double[mn * ld];
        foreach (ref v; a) v = rng.bipolar();
        foreach (ref v; b) v = rng.bipolar();
        foreach (ref v; c) v = rng.bipolar();
        foreach (layout; 0 .. 2)
        {
            immutable size_t am = layout ? 1 : sn, as = layout ? mn : 1;
            auto want = c.dup, got = c.dup;
            foreach (m; 0 .. mn)
                foreach (k; 0 .. kn)
                    foreach (s; 0 .. sn)
                        want[m * ld + k] += a[m * am + s * as] * b[s * ld + k];
            multiplyAdd(got, a, am, as, b, mn, sn, kn, ld);
            assert(got == want);
        }
    }
}
