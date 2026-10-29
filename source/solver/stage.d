/// A CMA-ES stage: CMA-ES over a problem from a start point, with a budget,
/// an early stop when the best has stalled, and optional pre-screening by a
/// surrogate model. Moved from pleaser's optimize.d (cmaStage), where every
/// CMA-ES stage of a run (cma1, cma2, ...) is one of these.
///
/// Deterministic: CMA-ES and the pre-screening draw from their own seeded
/// generators, the problem evaluates a batch as one, and the candidates are
/// taken in population order; so a seed gives the same stage on any thread
/// count.
///
/// A stage can stop after any generation and go on later (`StageRun`): it
/// then ends as the stage that never stopped, and a finished stage resumed
/// with a larger budget as the stage given that budget from the start.
module solver.stage;

import std.exception : enforce;

import solver.cmaes : CmaEs;
import solver.problem;
import solver.rng : SplitMix64;

/// What a pre-screened stage asks of its model: learn from evaluations,
/// train, predict. solver.model's LearnedModel is one.
interface StageModel(E)
{
    /// Adds evaluated candidates (genes in [0, 1]) to the training data.
    void learn(const(double[])[] genes, const(E)[] evals);
    void train(size_t epochs);
    /// Per candidate: the predicted objective, and how far over the hard
    /// limits it is predicted (with the model's headroom; 0 within them).
    void predictScore(const(double[])[] genes, double[] score, double[] violation);
}

struct StageOptions
{
    size_t lambda;            // population; the caller's choice (pleaser raises CMA-ES's default to its threads)
    double sigma;             // initial step size in gene units
    size_t budget;            // evaluations, rounded up to whole generations
    ulong  seed;
    /// Early stop: the stage ends when the best value has risen by less
    /// than stallTolerance over the last stallGenerations generations (0
    /// generations = never).
    size_t stallGenerations = 100;
    double stallTolerance = 0.01;
    /// Pre-screening (with a model): from generation surrogateFrom on, the
    /// stage samples surrogatePool times its population and evaluates a
    /// population's worth, surrogateShare of them the candidates the model
    /// ranks best, the rest at random; the model trains surrogateEpochs
    /// more before each of those generations.
    size_t surrogatePool = 0;
    double surrogateShare = 0.5;
    size_t surrogateFrom = 5;
    size_t surrogateEpochs = 10;
    Ranking ranking;
}

/// The best candidate so far, across stages, and the evaluations spent.
/// A stage starts from the caller's and updates it.
struct Best(E)
{
    E        eval;
    double[] genes;           // in [0, 1]
    bool     feasible;        // false: nothing within the limits yet (eval is then the caller's start)
    size_t   evals;           // evaluations spent so far, this stage's included
}

/// Whether `P` counts why candidates broke their limits.
enum bool hasCauses(P) = is(P.Causes) && is(typeof((const ref P cp, ref P.Causes c, const ref P.Eval e)
{
    P.Causes n = cp.newCauses();
    cp.countCauses(c, e);
}));

/// One generation, as `cmaStage` reports it after updating the best.
struct StageGeneration(P)
{
    size_t generation;        // from 1
    size_t before;            // evaluations of the run before this generation
    const(double[])[] genes;  // evaluated, clamped to [0, 1], in population order
    const(P.Eval)[] evals;
    double best;              // the best value so far (the caller's best included); -inf while none is feasible
    double populationBest;    // this generation's best feasible value; NaN when none
    size_t infeasible;        // candidates over a hard or terminal limit
    double sigma;             // CMA-ES's step size after this generation
    static if (hasCauses!P)
        P.Causes causes;      // why this generation's candidates broke their limits
    /// Pre-screened generations only (`picked` null otherwise): of the
    /// `sampled` candidates, the population indices evaluated (in the
    /// order of `genes`), which of the sampled ones the model chose, and
    /// how many of its choices it predicted within the limits.
    const(size_t)[] picked;
    const(bool)[] byModel;
    size_t sampled, predictedFeasible;
}

/// How a stage ended.
struct StageEnd
{
    size_t generations, spent;
    bool   stalled;           // stopped early (stallGenerations)
    double rise;              // ... the best's rise over the window it stalled in
    bool   stopped;           // stopped by the caller (afterGeneration), to go on later
}

/// A stage between generations, as plain data: with the caller's `Best`
/// and model (`LearnedModel.save`), what `cmaStage` needs to go on as the
/// stage that never stopped. Every field is a value or an array of values.
struct StageRun
{
    StageOptions options;     // the stage's; a resumed stage checks them (all but the budget and the stall test)
    size_t generation;        // generations done
    size_t spent;             // evaluations of this stage
    double[] history;         // the best value after each generation
    CmaEs.State es;           // CMA-ES, with its population (larger once pre-screening has begun)
    ulong rng;                // the pre-screening's generator
}

/// Runs CMA-ES on `p` from `x0` for `o.budget` evaluations, updating `best`
/// with every better feasible candidate. `screenModel` (null: none) pre-screens
/// as StageOptions says. `generationDone` is called after every generation.
///
/// With `run`, the stage keeps its state there after every generation, and
/// `afterGeneration` (null: never) is asked after each whether to stop: the
/// stage then returns with `stopped` set and `run` ready to save. Passed a
/// `run` that has generations done, the stage goes on from it instead of
/// starting from `x0` (only its length is checked), with the same options
/// but for the budget and the stall test, the caller's `best` and model as
/// they were when it stopped. The budget is the whole stage's. The stall
/// test is applied to the saved history at once: a resumed stage equals the
/// stage given these settings from the start unless they would have
/// stopped it before it was saved (a longer window or a smaller tolerance
/// never would). A model must be given when its run samples for one.
///
/// The problem gets the genes as CMA-ES sampled them: they may lie outside
/// [0, 1], and the problem must read them clamped (pleaser's decode does,
/// and one of its genes compares the raw value, so the stage leaves the
/// clamping to it). The fitness adds the ranking's bound penalty for how far
/// out they were; the best, the model and the report get them clamped.
StageEnd cmaStage(P, B, M)(ref P p, ref B best, const(double)[] x0, const ref StageOptions o, M screenModel,
                           scope void delegate(const ref StageGeneration!P) generationDone = null,
                           StageRun* run = null, scope bool delegate(const ref StageRun) afterGeneration = null)
    if (isProblem!P && is(B == Best!(P.Eval)) && (is(M == typeof(null)) || is(M : StageModel!(P.Eval))))
{
    // No model (null) or one, as a StageModel: so that the calls below
    // compile either way.
    StageModel!(P.Eval) model = screenModel;
    immutable bool haveModel = model !is null;
    enforce(o.lambda > 0, "a stage needs a population (StageOptions.lambda)");
    enforce(!haveModel || o.surrogatePool > 0, "a pre-screening model needs StageOptions.surrogatePool > 0");
    immutable size_t lambda = o.lambda;
    immutable bool resumed = run !is null && run.generation > 0;
    if (resumed)
    {
        StageOptions was = run.options;
        was.budget = o.budget;
        was.stallGenerations = o.stallGenerations;
        was.stallTolerance = o.stallTolerance;
        enforce(was == o, "a stage resumed goes on with its own options (all but the budget and the stall test)");
        enforce(run.es.mean.length == x0.length, "a stage resumed goes on in its own gene space");
        enforce(run.es.lambda == lambda || (haveModel && run.es.lambda == o.surrogatePool * lambda),
                "a pre-screened stage resumed after its pre-screening began needs its model");
        enforce(run.history.length == run.generation, "a stage resumed needs its history");
    }
    auto es = resumed ? CmaEs.restore(run.es) : CmaEs(x0, o.sigma, lambda, o.seed);
    auto fit = new double[lambda];
    auto rows = new double[][](lambda);
    size_t spent = resumed ? run.spent : 0, generation = resumed ? run.generation : 0;
    double[] history = resumed ? run.history.dup : null;     // best value after each generation
    // Pre-screening: every sampled candidate's predicted fitness, which are
    // evaluated, and which of those the model chose.
    double[] predicted;
    size_t[] picked;
    bool[] byModel;
    auto rng = SplitMix64(resumed ? run.rng : o.seed ^ 0x5C2E_E7ED_5C2E_E7EDUL);
    StageEnd end;

    // The stall test, after each generation and on a resumed history
    // before going on.
    bool stalledNow()
    {
        if (!stalled(history, o.stallGenerations, o.stallTolerance))
            return false;
        end.stalled = true;
        end.rise = history[$ - 1] - history[$ - 1 - o.stallGenerations];
        return true;
    }
    void keep()
    {
        run.options = o;
        run.generation = generation;
        run.spent = spent;
        run.history = history.dup;
        run.es = es.save();
        run.rng = rng.state;
    }

    immutable bool stalledBefore = resumed && stalledNow();
    while (spent < o.budget && !stalledBefore)
    {
        ++generation;
        if (haveModel && generation >= o.surrogateFrom)
        {
            model.train(o.surrogateEpochs);
            if (es.populationSize == lambda)    // from here on, sample the larger population
                es = CmaEs(es.currentMean, es.stepSize, o.surrogatePool * lambda, o.seed + 0x5C2E);
        }
        auto pop = es.ask();
        size_t sampled = 0, predictedFeasible = 0;
        if (pop.length == lambda)
        {
            picked = null;
            foreach (k, x; pop)
                rows[k] = x;
        }
        else
        {
            // Rank the whole population on the model; evaluate its best
            // share, the rest of lambda drawn at random from the others.
            auto score = new double[pop.length], viol = new double[pop.length];
            model.predictScore(pop, score, viol);
            predicted = new double[pop.length];
            foreach (k, x; pop)
            {
                static if (__traits(hasMember, P, "screenPenalty"))
                    immutable double value = score[k] - p.screenPenalty(x);
                else
                    immutable double value = score[k];
                predicted[k] = o.ranking.fitness(Breach(viol[k]), value, boundExcess(x));
            }
            size_t[] chosen;
            picked = pickRendered(predicted, lambda, o.surrogateShare, rng, chosen);
            byModel = new bool[pop.length];
            foreach (i; chosen)
                byModel[i] = true;
            foreach (j, i; picked)
                rows[j] = pop[i];
            sampled = pop.length;
            foreach (i; picked)
                predictedFeasible += byModel[i] && viol[i] == 0;
        }
        auto evals = p.evaluate(rows, Scoring.feasibleOnly);
        // The candidates as evaluated: the model learns, the report keeps
        // and the best takes them so.
        auto genes = new double[][lambda];
        foreach (k, x; rows)
            genes[k] = clamp01(x);
        if (haveModel)
            model.learn(genes, evals);
        immutable size_t before = best.evals;
        best.evals += lambda;
        spent += lambda;

        double genBest = double.nan;
        size_t infeasible = 0;
        static if (hasCauses!P)
            auto causes = p.newCauses();
        foreach (k, x; rows)
        {
            static if (hasCauses!P)
                p.countCauses(causes, evals[k]);
            immutable Breach b = p.breach(evals[k]);
            if (!b.feasible)
            {
                fit[k] = o.ranking.fitness(b, 0, boundExcess(x));
                ++infeasible;
                continue;
            }
            immutable double s = value(p.objective(evals[k]), b);
            fit[k] = o.ranking.fitness(b, s, boundExcess(x));
            genBest = !(genBest >= s) ? s : genBest;
            if (!best.feasible || s > valueOf(p, best.eval))
            {
                best.eval = evals[k];
                best.genes = genes[k].dup;
                best.feasible = true;
            }
        }
        if (picked is null)
            es.tell(fit);
        else
        {
            // The evaluated candidates with their real fitness, the rest
            // with the model's.
            auto all = predicted.dup;
            foreach (j, i; picked)
                all[i] = fit[j];
            es.tell(all);
        }

        immutable double bestNow = best.feasible ? valueOf(p, best.eval) : -double.infinity;
        if (generationDone !is null)
        {
            StageGeneration!P g;
            g.generation = generation;
            g.before = before;
            g.genes = genes;
            g.evals = evals;
            g.best = bestNow;
            g.populationBest = genBest;
            g.infeasible = infeasible;
            g.sigma = es.stepSize;
            static if (hasCauses!P)
                g.causes = causes;
            g.picked = picked;
            g.byModel = byModel;
            g.sampled = sampled;
            g.predictedFeasible = predictedFeasible;
            generationDone(g);
        }

        history ~= bestNow;
        if (run !is null)
            keep();
        if (stalledNow())
            break;
        if (afterGeneration !is null && run !is null && spent < o.budget && afterGeneration(*run))
        {
            end.stopped = true;
            break;
        }
    }
    if (run !is null && !resumed && generation == 0)
        keep();
    end.generations = generation;
    end.spent = spent;
    return end;
}

/// Which of a pre-screened population to evaluate, in index order: the
/// share `share` of `lambda` its `predicted` fitness ranks best (ties to the
/// lower index), then the rest of `lambda` drawn with `rng` (a SplitMix64)
/// from the others. `byModel`: the ones the model chose, best first.
size_t[] pickRendered(R)(const(double)[] predicted, size_t lambda, double share, ref R rng,
                         out size_t[] byModel) pure
{
    import std.algorithm : sort, SwapStrategy;
    auto order = new size_t[predicted.length];
    foreach (j, ref o; order)
        o = j;
    sort!((a, b) => predicted[a] < predicted[b] || (predicted[a] == predicted[b] && a < b), SwapStrategy.stable)(order);
    immutable size_t best = cast(size_t)(share * lambda + 0.5);
    auto picked = order[0 .. best].dup;
    byModel = picked.dup;
    auto others = order[best .. $].dup;
    sort(others);
    foreach (j; best .. lambda)
    {
        immutable size_t r = cast(size_t)(rng.next() % others.length);
        picked ~= others[r];
        others = others[0 .. r] ~ others[r + 1 .. $];
    }
    sort(picked);
    return picked;
}

/// Whether a stage has stalled: its best value (`history`, after each
/// generation) rose by less than `tolerance` over the last `generations`
/// (0: never).
bool stalled(const(double)[] history, size_t generations, double tolerance) pure nothrow @nogc @safe
{
    return generations > 0 && history.length > generations
        && history[$ - 1] - history[$ - 1 - generations] < tolerance;
}

version (unittest)
{
    /// A sphere with a hard limit: maximise -(x - 0.3)² summed, with
    /// x[0] ≤ 0.5 hard. Counts its evaluations.
    private struct Sphere
    {
        struct Eval { bool scored; double obj = double.nan; double over; }
        struct Causes
        {
            size_t total, over;
            void opOpAssign(string op : "+")(const Causes c) { total += c.total; over += c.over; }
        }

        size_t n = 4, calls;

        size_t dimension() const { return n; }

        Eval[] evaluate(const(double[])[] genes, Scoring scoring)
        {
            auto o = new Eval[genes.length];
            foreach (k, g; genes)
            {
                ++calls;
                const x = clamp01(g);
                o[k].over = x[0] > 0.5 ? x[0] - 0.5 : 0;
                o[k].scored = scoring == Scoring.all || o[k].over == 0;
                if (o[k].scored)
                {
                    double s = 0;
                    foreach (v; x)
                        s -= (v - 0.3) ^^ 2;
                    o[k].obj = s;
                }
            }
            return o;
        }

        bool   scored(const ref Eval e) const { return e.scored; }
        double objective(const ref Eval e) const { return e.obj; }
        Breach breach(const ref Eval e) const { return Breach(e.over); }

        Causes newCauses() const { return Causes.init; }
        void countCauses(ref Causes c, const ref Eval e) const { ++c.total; c.over += e.over > 0; }
    }

    static assert(isProblem!Sphere && hasCauses!Sphere);

    /// A model that predicts the sphere exactly, for pre-screening.
    private final class Exact : StageModel!(Sphere.Eval)
    {
        size_t learned, trained;
        void learn(const(double[])[] genes, const(Sphere.Eval)[] evals) { learned += genes.length; }
        void train(size_t epochs) { trained += epochs; }
        void predictScore(const(double[])[] genes, double[] score, double[] violation)
        {
            Sphere s;
            foreach (k, e; s.evaluate(genes, Scoring.all))
            {
                score[k] = e.obj;
                violation[k] = e.over;
            }
        }
    }
}

unittest
{
    // The stage finds the optimum within the limit, spends whole
    // generations and reports each one.
    Sphere p;
    Best!(Sphere.Eval) best;
    best.genes = [0.9, 0.9, 0.9, 0.9];
    StageOptions o = {lambda: 8, sigma: 0.2, budget: 400, seed: 3, stallGenerations: 0};
    size_t gens, last;
    double prev = -double.infinity;
    auto end = cmaStage(p, best, best.genes, o, null, (const ref StageGeneration!Sphere g) {
        ++gens;
        assert(g.generation == gens && g.before == last && g.evals.length == 8 && g.picked is null);
        assert(g.causes.total == 8 && g.causes.over == g.infeasible);
        assert(g.best >= prev);
        prev = g.best;
        last = g.before + 8;
    });
    assert(end.generations == 50 && end.spent == 400 && best.evals == 400 && p.calls == 400 && !end.stalled);
    assert(best.feasible && best.eval.obj > -1e-4);
    foreach (v; best.genes)
        assert(v >= 0.29 && v <= 0.31);
}

unittest
{
    // A stage needs a population, and a model a larger sample to screen.
    import std.exception : assertThrown;
    Sphere p;
    Best!(Sphere.Eval) best;
    best.genes = [0.5, 0.5, 0.5, 0.5];
    StageOptions o = {lambda: 0, sigma: 0.2, budget: 40, seed: 3};
    assertThrown(cmaStage(p, best, best.genes, o, null));
    o.lambda = 8;
    assertThrown(cmaStage(p, best, best.genes, o, new Exact));
}

unittest
{
    // The same seed gives the same stage; an early stop ends it.
    Sphere p1, p2;
    Best!(Sphere.Eval) b1, b2;
    StageOptions o = {lambda: 6, sigma: 0.1, budget: 3000, seed: 11, stallGenerations: 10, stallTolerance: 1e-3};
    immutable double[] x0 = [0.5, 0.5, 0.5, 0.5];
    auto e1 = cmaStage(p1, b1, x0, o, null);
    auto e2 = cmaStage(p2, b2, x0, o, null);
    assert(e1 == e2 && b1.genes == b2.genes && b1.eval == b2.eval);
    assert(e1.stalled && e1.spent < 3000 && e1.rise < 1e-3);
}

unittest
{
    // Pre-screened: from surrogateFrom on, a population's worth of a pool
    // four times larger, half the model's best.
    Sphere p;
    Best!(Sphere.Eval) best;
    auto m = new Exact;
    StageOptions o = {lambda: 8, sigma: 0.2, budget: 160, seed: 5, stallGenerations: 0,
                      surrogatePool: 4, surrogateShare: 0.5, surrogateFrom: 3, surrogateEpochs: 2};
    size_t screened;
    cmaStage(p, best, [0.8, 0.2, 0.8, 0.2], o, m, (const ref StageGeneration!Sphere g) {
        if (g.generation < 3)
            return assert(g.picked is null);
        ++screened;
        assert(g.sampled == 32 && g.picked.length == 8 && g.byModel.length == 32);
        size_t chosen;
        foreach (i; g.picked)
            chosen += g.byModel[i];
        assert(chosen == 4 && g.predictedFeasible <= 4);
    });
    assert(screened == 18 && m.learned == 160 && m.trained == 2 * 18 && p.calls == 160);
}

version (unittest)
{
    /// Every generation's numbers and genes, flat, to compare stages bit
    /// for bit.
    private double[] trace(const ref StageGeneration!Sphere g)
    {
        double[] t = [g.generation, g.before, g.best, g.populationBest, g.sigma, g.infeasible];
        foreach (x; g.genes)
            t ~= x;
        return t;
    }

    private bool same(const(double)[] a, const(double)[] b)
    {
        if (a.length != b.length)
            return false;
        foreach (i; 0 .. a.length)
            if (a[i] !is b[i])
                return false;
        return true;
    }

    /// A stage run whole, or stopped after each of `stops` (generations, in
    /// order) and resumed from its saved state each time: the trace, the
    /// end and the best.
    private struct Stopped
    {
        double[] trace;
        StageEnd end;
        Best!(Sphere.Eval) best;
        size_t calls;
        StageRun run;
    }

    private Stopped stopped(const ref StageOptions o, const(size_t)[] stops, Exact model = null)
    {
        Stopped r;
        Sphere p;
        r.best.genes = [0.8, 0.2, 0.8, 0.2];
        immutable double[] x0 = r.best.genes.idup;
        size_t next = 0;
        void done(const ref StageGeneration!Sphere g) { r.trace ~= trace(g); }
        bool stop(const ref StageRun s) { return next < stops.length && s.generation == stops[next]; }
        do
        {
            r.end = model is null ? cmaStage(p, r.best, x0, o, null, &done, &r.run, &stop)
                                  : cmaStage(p, r.best, x0, o, model, &done, &r.run, &stop);
            if (r.end.stopped)
            {
                // Saved and loaded: the run as plain data, copied.
                StageRun copy = r.run;
                copy.history = r.run.history.dup;
                copy.es = CmaEs.restore(r.run.es).save();
                r.run = copy;
                ++next;
            }
        }
        while (r.end.stopped);
        r.calls = p.calls;
        return r;
    }
}

unittest
{
    // Stopped after any generations and resumed, a stage ends as the one
    // that never stopped, bit for bit.
    StageOptions o = {lambda: 8, sigma: 0.2, budget: 400, seed: 3, stallGenerations: 0};
    auto whole = stopped(o, null);
    assert(!whole.end.stopped && whole.end.generations == 50 && whole.run.generation == 50);
    foreach (stops; [[1UL], [7UL], [49UL], [3UL, 4, 20, 48]])
    {
        auto r = stopped(o, stops);
        assert(same(r.trace, whole.trace) && r.end is whole.end && r.calls == whole.calls);
        assert(r.best.genes == whole.best.genes && r.best.eval == whole.best.eval && r.best.evals == whole.best.evals);
    }

    // A finished stage resumed with a larger budget ends as the stage given
    // that budget from the start.
    StageOptions longer = o;
    longer.budget = 800;
    auto full = stopped(longer, null);
    Sphere p;
    auto best = whole.best;
    best.genes = whole.best.genes.dup;
    auto run = whole.run;
    double[] t = whole.trace.dup;
    const end = cmaStage(p, best, [0.8, 0.2, 0.8, 0.2], longer, null,
                         (const ref StageGeneration!Sphere g) { t ~= trace(g); }, &run);
    assert(same(t, full.trace) && end is full.end && best.genes == full.best.genes && run.generation == 100);
}

unittest
{
    // Pre-screened: stopped before and after the pre-screening began, and
    // resumed with the model, the same stage.
    StageOptions o = {lambda: 8, sigma: 0.2, budget: 240, seed: 5, stallGenerations: 0,
                      surrogatePool: 4, surrogateShare: 0.5, surrogateFrom: 3, surrogateEpochs: 2};
    auto m1 = new Exact, m2 = new Exact;
    auto whole = stopped(o, null, m1);
    auto r = stopped(o, [1, 2, 3, 10], m2);
    assert(same(r.trace, whole.trace) && r.end is whole.end && r.best.genes == whole.best.genes);
    assert(m1.learned == m2.learned && m1.trained == m2.trained);
    assert(whole.run.es.lambda == 32);
}

unittest
{
    // The stall test on a resumed stage: a stalled stage stays stopped with
    // its own settings; with a longer window it goes on as the stage given
    // that window from the start.
    StageOptions o = {lambda: 6, sigma: 0.1, budget: 3000, seed: 11, stallGenerations: 10, stallTolerance: 1e-3};
    auto first = stopped(o, null);
    assert(first.end.stalled && first.end.spent < 3000);
    Sphere p;
    auto best = first.best;
    auto run = first.run;
    auto again = cmaStage(p, best, [0.8, 0.2, 0.8, 0.2], o, null, null, &run);
    assert(again.stalled && again.generations == first.end.generations && p.calls == 0);

    StageOptions wider = o;
    wider.stallGenerations = 40;
    auto whole = stopped(wider, null);
    assert(whole.end.generations > first.end.generations);
    double[] t = first.trace.dup;
    best = first.best;
    best.genes = first.best.genes.dup;
    run = first.run;
    run.history = first.run.history.dup;
    const end = cmaStage(p, best, [0.8, 0.2, 0.8, 0.2], wider, null,
                         (const ref StageGeneration!Sphere g) { t ~= trace(g); }, &run);
    assert(same(t, whole.trace) && end is whole.end && best.genes == whole.best.genes);
}

unittest
{
    // A stage resumed with other options, without the model it samples
    // for, or in another gene space, refuses to.
    import std.exception : assertThrown;
    StageOptions o = {lambda: 8, sigma: 0.2, budget: 160, seed: 5, stallGenerations: 0,
                      surrogatePool: 4, surrogateShare: 0.5, surrogateFrom: 3, surrogateEpochs: 2};
    Sphere p;
    Best!(Sphere.Eval) best;
    StageRun run;
    cmaStage(p, best, [0.5, 0.5, 0.5, 0.5], o, new Exact, null, &run, (const ref StageRun s) => s.generation == 5);
    StageOptions other = o;
    other.seed = 6;
    auto copy = run;
    assertThrown(cmaStage(p, best, [0.5, 0.5, 0.5, 0.5], other, new Exact, null, &copy));
    StageOptions plain = o;
    plain.surrogatePool = 0;
    copy = run;
    copy.options.surrogatePool = 0;
    assertThrown(cmaStage(p, best, [0.5, 0.5, 0.5, 0.5], plain, null, null, &copy));
    copy = run;
    assertThrown(cmaStage(p, best, [0.5, 0.5, 0.5], o, new Exact, null, &copy));
}

unittest
{
    import std.algorithm : sort;
    // pickRendered: the share by the model, the rest at random, index order.
    auto rng = SplitMix64(1);
    size_t[] byModel;
    auto picked = pickRendered([5.0, 1, 4, 2, 3, 0, 6, 7], 4, 0.5, rng, byModel);
    assert(byModel == [5, 1] && picked.length == 4 && picked == picked.dup.sort.release);
    assert(stalled([1, 1.001, 1.002], 2, 0.01) && !stalled([1, 1.5, 2], 2, 0.01) && !stalled([1, 1], 0, 1));
}
