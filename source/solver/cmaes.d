/// CMA-ES (Hansen's (mu/mu_w, lambda) evolution strategy with rank-one and
/// rank-mu covariance updates and cumulative step-size adaptation), for
/// minimisation. Follows "The CMA Evolution Strategy: A Tutorial" (Hansen,
/// 2016), with the default parameter settings from its Table 1.
///
/// Deterministic: a fixed RNG (SplitMix64 with Box-Muller), a stable sort on
/// fitness with ties broken by index, and a Jacobi eigensolver with a fixed
/// sweep order, so a seed gives the same run on any machine or thread count.
module solver.cmaes;

import std.algorithm : sort, SwapStrategy;
import std.math : cos, exp, log, sqrt, PI, abs;

import solver.rng : SplitMix64;

struct CmaEs
{
    /// `x0`: start point; `sigma0`: initial step size (in gene units);
    /// `lambda`: population size, 0 for the default 4 + floor(3 ln n).
    this(const(double)[] x0, double sigma0, size_t lambda, ulong seed) pure nothrow
    {
        p = Params.of(x0.length, lambda);
        immutable size_t n = p.n;
        st.lambda = p.lambda;
        st.sigma = sigma0;
        st.rng = seed;
        st.mean = x0.dup;
        st.pc = new double[n];
        st.pc[] = 0;
        st.ps = new double[n];
        st.ps[] = 0;
        st.D = new double[n];
        st.D[] = 1;
        st.C = identity(n);
        st.B = identity(n);
        X = new double[][](p.lambda, n);
        Y = new double[][](p.lambda, n);
        z = new double[n];
    }

    /// The next population, lambda rows of n. Valid until the next ask().
    double[][] ask() pure nothrow @nogc
    {
        with (st)
            foreach (k; 0 .. p.lambda)
            {
                foreach (i; 0 .. p.n)
                    z[i] = D[i] * gaussian();
                foreach (i; 0 .. p.n)
                {
                    double y = 0;
                    foreach (j; 0 .. p.n)
                        y += B[i][j] * z[j];
                    Y[k][i] = y;
                    X[k][i] = mean[i] + sigma * y;
                }
            }
        return X;
    }

    /// Updates the distribution from the fitness of the last ask()'s rows
    /// (lower is better).
    void tell(const(double)[] fitness) pure
    {
        assert(fitness.length == p.lambda);
        immutable size_t n = p.n, mu = p.mu;
        const w = p.w;
        auto order = new size_t[p.lambda];
        foreach (i, ref o; order) o = i;
        sort!((a, b) => fitness[a] < fitness[b] || (fitness[a] == fitness[b] && a < b),
              SwapStrategy.stable)(order);

        with (st)
        {
            if (fitness[order[0]] < bestF)
            {
                bestF = fitness[order[0]];
                bestX = X[order[0]].dup;
            }

            // Weighted mean step in y-space, and the new mean.
            auto yw = new double[n];
            yw[] = 0;
            foreach (r; 0 .. mu)
                foreach (i; 0 .. n)
                    yw[i] += w[r] * Y[order[r]][i];
            foreach (i; 0 .. n)
                mean[i] += sigma * yw[i];

            // ps: conjugate evolution path, C^-1/2 yw = B D^-1 B' yw.
            auto tmp = new double[n];
            foreach (j; 0 .. n)
            {
                double s = 0;
                foreach (i; 0 .. n) s += B[i][j] * yw[i];
                tmp[j] = s / D[j];
            }
            immutable double csn = sqrt(p.cs * (2 - p.cs) * p.mueff);
            foreach (i; 0 .. n)
            {
                double s = 0;
                foreach (j; 0 .. n) s += B[i][j] * tmp[j];
                ps[i] = (1 - p.cs) * ps[i] + csn * s;
            }
            immutable double psNorm = norm(ps);
            ++gen;
            immutable bool hsig = psNorm / sqrt(1 - (1 - p.cs) ^^ (2 * gen)) / p.chiN < 1.4 + 2 / (n + 1.0);

            immutable double ccn = sqrt(p.cc * (2 - p.cc) * p.mueff);
            foreach (i; 0 .. n)
                pc[i] = (1 - p.cc) * pc[i] + (hsig ? ccn * yw[i] : 0);

            // Covariance: rank-one and rank-mu updates, upper triangle mirrored.
            immutable double keep = 1 - p.c1 - p.cmu + (hsig ? 0 : p.c1 * p.cc * (2 - p.cc));
            foreach (i; 0 .. n)
                foreach (j; i .. n)
                {
                    double rmu = 0;
                    foreach (r; 0 .. mu)
                        rmu += w[r] * Y[order[r]][i] * Y[order[r]][j];
                    immutable double v = keep * C[i][j] + p.c1 * pc[i] * pc[j] + p.cmu * rmu;
                    C[i][j] = v;
                    C[j][i] = v;
                }

            sigma *= exp((p.cs / p.damps) * (psNorm / p.chiN - 1));
        }
        decompose();
    }

    /// Everything that changes as it runs, for a checkpoint: the constants
    /// follow from the dimension and lambda (Params), and ask()'s rows are
    /// scratch. Taken between tell() and the next ask(), restore()
    /// continues bit-identically.
    struct State
    {
        size_t lambda;
        double sigma;
        long gen;
        ulong rng;                // the generator's state
        double bestF = double.infinity;
        bool haveSpare;
        double spare;
        double[] mean, pc, ps, D, bestX;
        double[][] C, B;
    }

    State save() const pure nothrow { return deepCopy(st); }

    static CmaEs restore(const ref State s) pure nothrow
    {
        auto e = CmaEs(s.mean, s.sigma, s.lambda, s.rng);
        assert(e.p.lambda == s.lambda && s.pc.length == e.p.n && s.C.length == e.p.n && s.B.length == e.p.n);
        e.st = deepCopy(s);
        return e;
    }

    double stepSize() const pure nothrow @nogc { return st.sigma; }
    const(double)[] currentMean() const pure nothrow @nogc { return st.mean; }
    size_t populationSize() const pure nothrow @nogc { return p.lambda; }
    size_t generation() const pure nothrow @nogc { return cast(size_t) st.gen; }
    double bestFitness() const pure nothrow @nogc { return st.bestF; }
    const(double)[] bestPoint() const pure nothrow @nogc { return st.bestX; }

private:
    /// The strategy's constants (the tutorial's Table 1), fixed by the
    /// dimension and the population size.
    struct Params
    {
        size_t n, lambda, mu;
        immutable(double)[] w;    // the recombination weights, summing to 1
        double mueff, cc, cs, c1, cmu, damps, chiN;

        /// `lambda` 0: the default 4 + floor(3 ln n).
        static Params of(size_t n, size_t lambda) pure nothrow
        {
            Params p;
            p.n = n;
            p.lambda = lambda ? lambda : 4 + cast(size_t)(3 * log(cast(double) n));
            p.mu = p.lambda / 2;
            auto w = new double[p.mu];
            double sw = 0;
            foreach (i; 0 .. p.mu) { w[i] = log(p.mu + 0.5) - log(i + 1.0); sw += w[i]; }
            double sw2 = 0;
            foreach (ref x; w) { x /= sw; sw2 += x * x; }
            p.w = w.idup;
            p.mueff = 1 / sw2;

            immutable double dn = n;
            p.cc    = (4 + p.mueff / dn) / (dn + 4 + 2 * p.mueff / dn);
            p.cs    = (p.mueff + 2) / (dn + p.mueff + 5);
            p.c1    = 2 / ((dn + 1.3) ^^ 2 + p.mueff);
            immutable double cm = 2 * (p.mueff - 2 + 1 / p.mueff) / ((dn + 2) ^^ 2 + p.mueff);
            p.cmu   = cm < 1 - p.c1 ? cm : 1 - p.c1;
            immutable double t = sqrt((p.mueff - 1) / (dn + 1)) - 1;
            p.damps = 1 + 2 * (t > 0 ? t : 0) + p.cs;
            p.chiN  = sqrt(dn) * (1 - 1 / (4 * dn) + 1 / (21 * dn * dn));
            return p;
        }
    }

    Params p;
    State st;
    double[][] X, Y;              // ask()'s rows, and their steps before the mean and sigma
    double[] z;

    double gaussian() pure nothrow @nogc
    {
        with (st)
        {
            if (haveSpare) { haveSpare = false; return spare; }
            auto g = SplitMix64(rng);
            double u1 = g.uniform01();
            if (u1 < 1e-300) u1 = 1e-300;
            immutable double u2 = g.uniform01();
            rng = g.state;
            immutable double r = sqrt(-2 * log(u1));
            spare = r * cos(2 * PI * u2 + PI / 2);   // = r sin(2 pi u2), via cos for one function
            haveSpare = true;
            return r * cos(2 * PI * u2);
        }
    }

    /// B, D from C = B diag(D^2) B'.
    void decompose() pure nothrow
    {
        auto ev = new double[p.n];
        jacobiEigen(st.C, st.B, ev);
        foreach (i; 0 .. p.n)
            st.D[i] = sqrt(ev[i] > 1e-20 ? ev[i] : 1e-20);
    }

    /// A State with arrays of its own.
    static State deepCopy(const ref State s) pure nothrow
    {
        State o;
        static foreach (i; 0 .. State.tupleof.length)
        {{
            alias T = typeof(State.tupleof[i]);
            static if (is(T == double[][]))
            {
                o.tupleof[i] = new double[][](s.tupleof[i].length);
                foreach (r, row; s.tupleof[i])
                    o.tupleof[i][r] = row.dup;
            }
            else static if (is(T == double[]))
                o.tupleof[i] = s.tupleof[i].dup;
            else
                o.tupleof[i] = s.tupleof[i];
        }}
        return o;
    }

    static double norm(const(double)[] v) pure nothrow @nogc
    {
        double s = 0;
        foreach (x; v) s += x * x;
        return sqrt(s);
    }

    static double[][] identity(size_t n) pure nothrow
    {
        auto m = new double[][](n, n);
        foreach (i; 0 .. n) { m[i][] = 0; m[i][i] = 1; }
        return m;
    }
}

/// Eigen-decomposition of the symmetric matrix `a` by cyclic Jacobi
/// rotations: `vectors` gets the eigenvectors as columns, `values` the
/// eigenvalues. `a` is left unchanged.
///
/// The work is kept in flat arrays, the eigenvectors one to a row (they
/// are transposed into `vectors` at the end): so the rotations of the rows
/// and of the eigenvectors run along contiguous memory and vectorise. Each
/// element goes through the same operations in the same order whatever the
/// layout, so the results are the same bits.
void jacobiEigen(const(double[])[] a, double[][] vectors, double[] values) pure nothrow
{
    import std.math : abs;
    immutable size_t n = a.length;
    auto m = new double[n * n];
    auto vt = new double[n * n];               // vt[p * n + k] = vectors[k][p]
    vt[] = 0;
    foreach (i; 0 .. n)
    {
        m[i * n .. (i + 1) * n] = a[i][];
        vt[i * n + i] = 1;
    }
    foreach (sweep; 0 .. 100)
    {
        double off = 0, diag = 0;
        foreach (i; 0 .. n)
        {
            diag += m[i * n + i] * m[i * n + i];
            foreach (j; i + 1 .. n)
                off += m[i * n + j] * m[i * n + j];
        }
        if (off <= 1e-30 * diag)
            break;
        foreach (p; 0 .. n)
            foreach (q; p + 1 .. n)
            {
                immutable double apq = m[p * n + q];
                if (abs(apq) < 1e-300)
                    continue;
                immutable double theta = (m[q * n + q] - m[p * n + p]) / (2 * apq);
                immutable double t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + sqrt(theta * theta + 1));
                immutable double c = 1 / sqrt(t * t + 1), s = t * c;
                for (size_t i = 0; i < n * n; i += n)   // columns p, q
                {
                    immutable double mkp = m[i + p], mkq = m[i + q];
                    m[i + p] = c * mkp - s * mkq;
                    m[i + q] = s * mkp + c * mkq;
                }
                rotate(m[p * n .. (p + 1) * n], m[q * n .. (q + 1) * n], c, s);    // rows p, q
                rotate(vt[p * n .. (p + 1) * n], vt[q * n .. (q + 1) * n], c, s);  // eigenvectors p, q
            }
    }
    foreach (i; 0 .. n)
    {
        values[i] = m[i * n + i];
        foreach (k; 0 .. n)
            vectors[k][i] = vt[i * n + k];
    }
}

/// (x, y) = (c x - s y, s x + c y), element by element.
private void rotate(double[] x, double[] y, double c, double s) pure nothrow @nogc
{
    assert(x.length == y.length);
    foreach (k; 0 .. x.length)
    {
        immutable double xk = x[k], yk = y[k];
        x[k] = c * xk - s * yk;
        y[k] = s * xk + c * yk;
    }
}

unittest
{
    // Eigen-decomposition reproduces the matrix.
    double[][] a = [[4.0, 1, 0.5], [1.0, 3, 0.2], [0.5, 0.2, 1]];
    auto v = new double[][](3, 3);
    auto e = new double[3];
    jacobiEigen(a, v, e);
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 3)
        {
            double s = 0;
            foreach (k; 0 .. 3) s += v[i][k] * e[k] * v[j][k];
            assert(abs(s - a[i][j]) < 1e-12);
        }
}

unittest
{
    // Solves a shifted, badly scaled ellipsoid in 10-D.
    enum size_t n = 10;
    double f(const(double)[] x)
    {
        double s = 0;
        foreach (i; 0 .. n)
            s += (10.0 ^^ (3.0 * i / (n - 1))) * (x[i] - 0.3) ^^ 2;
        return s;
    }
    auto x0 = new double[n];
    x0[] = 0.8;
    auto es = CmaEs(x0, 0.3, 0, 42);
    auto fit = new double[es.populationSize];
    foreach (g; 0 .. 400)
    {
        auto pop = es.ask();
        foreach (k, x; pop) fit[k] = f(x);
        es.tell(fit);
        if (es.bestFitness < 1e-12) break;
    }
    assert(es.bestFitness < 1e-10);
}

unittest
{
    // A restored state continues bit-identically, from mid-run, with a
    // Gaussian spare pending (an odd dimension leaves one).
    enum size_t n = 7;
    double f(const(double)[] x)
    {
        double s = 0;
        foreach (i, v; x)
            s += (i + 1) * (v - 0.2) ^^ 2;
        return s;
    }
    auto x0 = new double[n];
    x0[] = 0.6;
    auto a = CmaEs(x0, 0.2, 0, 9);
    auto fit = new double[a.populationSize];
    void step(ref CmaEs es)
    {
        foreach (k, x; es.ask())
            fit[k] = f(x);
        es.tell(fit);
    }
    foreach (g; 0 .. 5)
        step(a);
    auto st = a.save();
    assert(st.haveSpare);
    auto b = CmaEs.restore(st);
    foreach (g; 0 .. 6)
    {
        step(a);
        step(b);
    }
    assert(a.currentMean == b.currentMean && a.stepSize == b.stepSize && a.bestFitness == b.bestFitness
           && a.save() == b.save());
}
