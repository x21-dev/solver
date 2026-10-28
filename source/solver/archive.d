/// The archive: the best candidate in each cell of a grid over a problem's
/// descriptors, found by CMA-MAE. Moved from pleaser's archive.d, where it
/// fills pleaser's character cells.
///
/// CMA-MAE (Fontaine and Nikolaidis, "Covariance Matrix Adaptation
/// MAP-Annealing", 2023): CMA-ES emitters whose candidates are ranked by
/// how far their value is above their cell's threshold. A cell's threshold
/// starts at the archive's minimum score and, each time a candidate beats
/// it, moves towards that candidate's value by the archive learning rate
/// alpha. At alpha 0 the emitters optimise the value alone; at alpha 1 the
/// threshold is the cell's best value, which is CMA-ME: only a better
/// candidate or a new cell counts. In between, a cell is worth exploring
/// until its threshold has caught up with what it holds. The archive itself
/// keeps the best candidate ever found in each cell (the paper's result
/// archive).
///
/// Infeasible candidates never enter the archive; emitters rank them below
/// every feasible candidate, by how far over the limits they are. An
/// improver restarts from a random elite when a few generations add
/// nothing, or are nearly all infeasible. Explorers (the last emitters)
/// head for empty cells instead, down the grid's distance, and take a new
/// target when they reach theirs or get stuck.
///
/// With a model (`ArchiveSurrogate`), explorers and optionally improvers
/// sample a larger population and render the share the model ranks best
/// (the rest at random); an explorer may also start where the model
/// predicts nearest its target.
///
/// Deterministic: emitter k draws from its own seed, every emitter's
/// population is evaluated in one batch, and the candidates enter the
/// archive in a fixed order (emitter by emitter, in population order), so
/// a seed gives the same archive on any thread count.
///
/// A run is plain data between iterations (`ArchiveRun`): saved and
/// restored field by field, it goes on as the run that never stopped
/// (`runArchive`'s `resume`).
module solver.archive;

import std.exception : enforce;

import solver.cmaes : CmaEs;
import solver.problem;
import solver.rng : SplitMix64;
import solver.stage : hasCauses, pickRendered;

/// Restart rules: an improver restarts after this many generations in a
/// row in which no candidate beat its cell's threshold; any emitter when
/// its last this many generations were at least RESTART_INFEASIBLE
/// infeasible, or its step size fell below RESTART_SIGMA.
enum size_t RESTART_GENERATIONS = 5;
enum double RESTART_INFEASIBLE = 0.9;
enum double RESTART_SIGMA = 1e-4;
/// Explorers: an explorer gives its target up after this many generations
/// without getting closer by EXPLORE_PROGRESS, and that cell is skipped for
/// EXPLORE_BLOCK iterations.
enum size_t EXPLORE_STALE = 10;
enum double EXPLORE_PROGRESS = 0.01;
enum size_t EXPLORE_BLOCK = 30;
/// Model-aimed explorers: points sampled around each elite, at the
/// explorers' step size.
enum size_t AIM_SAMPLES = 200;

// ─────────────────────────────────────────────────────────────────────────────
// The archive
// ─────────────────────────────────────────────────────────────────────────────

struct Elite(E)
{
    bool     occupied;
    double[] genes;           // in [0, 1]
    E        eval;
    size_t   found;           // evaluations spent when it entered
    double   value = 0;       // what it ranks by (valueOf)
}

/// What offering a candidate to the archive did.
struct Offer
{
    double value;             // its value minus its cell's threshold before it: ranks it for its emitter
    bool improved;            // it beat the threshold (which moved)
    bool newCell;             // it opened its cell
}

/// Cells, their thresholds and their best candidates. With `N` cells fixed
/// at compile time, held by value (`Elite!E[N]`: a copy is a copy); with
/// `N` 0, in arrays of `cells` given to the constructor. Thresholds start
/// at `minScore`, the value's lower bound.
struct Archive(E, size_t N = 0, double minScore = 0)
{
    static if (N)
    {
        Elite!E[N]  elites;
        double[N]   threshold = minScore;
    }
    else
    {
        Elite!E[]   elites;
        double[]    threshold;

        this(size_t cells) pure nothrow
        {
            elites = new Elite!E[cells];
            threshold = new double[cells];
            threshold[] = minScore;
        }
    }
    double alpha = 0.1;

    /// Offers a feasible candidate of value `f` in `cell`: moves the cell's
    /// threshold when it beats it, and keeps the candidate when it is the
    /// cell's best so far. Its value is negative when below the threshold.
    Offer offer(size_t cell, double f, const(double)[] genes, const ref E e, size_t found) pure nothrow
    {
        immutable double t = threshold[cell];
        immutable bool improved = f > t, newCell = !elites[cell].occupied;
        if (improved)
            threshold[cell] = (1 - alpha) * t + alpha * f;
        if (newCell || f > elites[cell].value)
            elites[cell] = Elite!E(true, genes.dup, copyOf(e), found, f);
        return Offer(f - t, improved, newCell);
    }

    size_t coverage() const pure nothrow @nogc
    {
        size_t n = 0;
        foreach (ref el; elites)
            n += el.occupied;
        return n;
    }

    /// The sum of the elites' values above minScore.
    double qdScore() const pure nothrow @nogc
    {
        double s = 0;
        foreach (ref el; elites)
            if (el.occupied)
                s += el.value - minScore;
        return s;
    }

    /// The best elite's value; the archive must hold one.
    double bestScore() const pure nothrow @nogc
    {
        assert(best != size_t.max);
        return elites[best].value;
    }

    /// The best elite's cell; size_t.max when empty. Ties: the lower index.
    size_t best() const pure nothrow @nogc
    {
        size_t b = size_t.max;
        foreach (c, ref el; elites)
            if (el.occupied && (b == size_t.max || el.value > elites[b].value))
                b = c;
        return b;
    }

    /// Occupied cells, in index order.
    size_t[] occupiedCells() const pure nothrow
    {
        size_t[] o;
        foreach (c, ref el; elites)
            if (el.occupied)
                o ~= c;
        return o;
    }
}

/// A copy of `e` that shares nothing with it: itself when its type holds
/// no mutable references, else with its arrays duplicated (one level).
E copyOf(E)(const ref E e) pure nothrow
{
    static if (is(const(E) : E))
        return e;
    else
    {
        E c;
        foreach (i, ref v; c.tupleof)
        {
            static if (is(typeof(v) : const(T)[], T))
                v = e.tupleof[i].dup;
            else
                v = e.tupleof[i];
        }
        return c;
    }
}

/// The elites to show: the best, then again and again the one farthest
/// (the grid's binDistance2) from those already picked, among the elites at
/// most `within` below the best; ties go to the lower cell index. At most
/// `count`.
size_t[] selectDiverse(A, G)(const ref A a, const ref G grid, double within, size_t count) pure
    if (isGrid!G)
{
    immutable size_t b = a.best;
    if (b == size_t.max)
        return null;
    immutable double floor = a.bestScore - within;
    size_t[] pool;
    foreach (c; a.occupiedCells)
        if (c != b && a.elites[c].value >= floor)
            pool ~= c;
    size_t[] picked = [b];
    while (picked.length < count && pool.length)
    {
        size_t pick = 0;
        long pickD = -1;
        foreach (i, c; pool)
        {
            long d = long.max;
            foreach (p; picked)
            {
                immutable long dd = grid.binDistance2(c, p);
                d = dd < d ? dd : d;
            }
            if (d > pickD)            // pool is in index order: ties keep the lower
            {
                pick = i;
                pickD = d;
            }
        }
        picked ~= pool[pick];
        pool = pool[0 .. pick] ~ pool[pick + 1 .. $];
    }
    return picked;
}

// ─────────────────────────────────────────────────────────────────────────────
// A run
// ─────────────────────────────────────────────────────────────────────────────

/// The options runArchive reads. A problem may bring its own options type
/// with these fields and more (pleaser's adds its processing cost), and an
/// optional `Ranking ranking`; `Ranking.init` without one.
struct ArchiveOptions
{
    size_t evals = 8000;      // budget, rounded up to whole iterations
    size_t emitters = 5;
    size_t lambda = 0;        // per emitter; 0 = CMA-ES default for the dimension
    double sigma = 0.05;      // emitters' initial step size, gene units
    double alpha = 0.1;       // archive learning rate
    ulong  seed = 1;
    /// How many of the emitters are explorers (the last ones), heading for
    /// empty cells instead of ranking by the thresholds; 0 = none.
    size_t explorers = 2;
    double explorerSigma = 0.1;
    /// Model-guided explorers: with a model, an explorer samples this many
    /// times its population and renders a population's worth:
    /// `surrogateShare` of them the ones the model ranks best, the rest
    /// drawn at random from the others; 0 = off. The model trains before
    /// every iteration from `surrogateFrom` on, `surrogateEpochs` more each
    /// time; until then explorers explore unguided.
    size_t surrogatePool = 0;
    double surrogateShare = 0.5;
    size_t surrogateFrom = 5;
    size_t surrogateEpochs = 10;
    /// Pre-screened improvers: the improvers sample and render as the
    /// model-guided explorers do, ranked on the model by how far their
    /// predicted value is above their predicted cell's threshold.
    bool surrogateImprovers = false;
    /// Model-aimed explorers: an explorer aimed at a new target once the
    /// model is ready starts where the model predicts nearest the target
    /// within the limits, of AIM_SAMPLES points around each elite (and the
    /// elites), instead of from the elite nearest it in bins.
    bool surrogateAim = false;
}

/// What a model gives the archive: it learns every evaluation, trains
/// between iterations, and predicts candidates it hasn't evaluated: for
/// row i of `genes`, its objective, breach (with whatever headroom the
/// problem wants on a prediction) and descriptors, at
/// `descriptors[i * d .. (i + 1) * d]` for the problem's d descriptors.
interface ArchiveSurrogate(E)
{
    void learn(const(double[])[] genes, const(E)[] evals);
    void train(size_t epochs);
    void predictArchive(const(double[])[] genes, double[] objective, Breach[] breach, double[] descriptors);
}

/// How the model-guided explorers' picks turned out: of the candidates
/// rendered, how many the model put within the limits and in the target,
/// and how many were.
struct GuideStats
{
    size_t sampled, rendered, random;   // rendered: the model's picks; random: the others rendered
    size_t predictedFeasible, feasible;
    size_t predictedInside, inside;
    size_t aims;                        // targets the model chose the start for (surrogateAim)
}

/// How the pre-screened improvers' picks turned out: of the model's picks
/// rendered, how many it put within the limits and above their cell's
/// threshold, and how many were.
struct ScreenStats
{
    size_t sampled, rendered, random;
    size_t predictedFeasible, feasible;
    size_t predictedImproving, improving;
}

/// Why an emitter starts again after a generation. An improver restarts
/// from an elite; an explorer heading for a target takes a new one.
enum RestartReason
{
    none,                     // it doesn't
    idle,                     // improver: no candidate beat its cell's threshold for RESTART_GENERATIONS generations
    infeasible,               // its last RESTART_GENERATIONS generations were nearly all over the limits
    converged,                // its step size collapsed (below RESTART_SIGMA)
    reached,                  // explorer: its target holds an elite now
    stuck,                    // explorer: no closer to its target for EXPLORE_STALE generations
}

/// A restart reason by its name, "" for none.
string reasonText(RestartReason r) pure nothrow @nogc @safe
{
    static immutable string[] names = [__traits(allMembers, RestartReason)];
    return r == RestartReason.none ? "" : names[r];
}

/// The cause counts of a problem without them.
struct NoCauses
{
    void opOpAssign(string op : "+")(const NoCauses) pure nothrow @nogc @safe {}
}

/// `P.Causes`, or NoCauses.
template CausesOf(P)
{
    static if (hasCauses!P)
        alias CausesOf = P.Causes;
    else
        alias CausesOf = NoCauses;
}

/// One emitter generation, for progress reports.
struct ArchiveGeneration(C)
{
    size_t iteration, emitter, evals;
    size_t cells;             // the archive's coverage after this generation
    double qdScore, bestScore;
    size_t improved, newCells;   // candidates that beat their threshold, that opened a cell
    double sigma;
    RestartReason restartReason;   // why the emitter restarts or retargets after this generation
    C      causes;
    bool   explorer;
    size_t target = size_t.max;   // explorer: the cell it heads for (size_t.max: none)
    double distance = double.nan; // explorer: this generation's closest feasible candidate to it
    size_t focus = size_t.max;    // focused run: the focused cell the emitter last started from
}

/// An emitter between iterations.
struct Emitter
{
    CmaEs es;
    SplitMix64 rng;           // picks restart elites
    size_t idle, restarts;
    size_t[] infeasible;      // per generation since the last (re)start
    bool explorer;
    size_t target = size_t.max;           // explorer: the empty cell it heads for
    double closest = double.infinity;     // ... its closest feasible candidate so far
    size_t stale;                         // ... generations without getting closer
    size_t from = size_t.max;             // focused run: the focused cell it last started from
}

/// A run of the archive: the result, and between iterations everything
/// needed to continue it exactly (the emitters, the cells the explorers are
/// blocked from).
struct ArchiveRun(P, O = ArchiveOptions, size_t N = 0, double minScore = 0)
{
    alias Eval = P.Eval;
    alias Causes = CausesOf!P;

    Archive!(P.Eval, N, minScore) archive;
    O options;
    size_t lambda, evals, iterations;
    size_t seedEvals;         // evaluations of the seeds, before the emitters
    Causes causes;            // over every candidate
    size_t restartsIdle, restartsInfeasible, restartsConverged;
    size_t[] reached;         // targets the explorers reached (or saw filled), in order
    size_t[] abandoned;       // targets they gave up on, in order
    P.Eval[] seeds;           // the seeds' evaluations, in the order given (none when resumed)
    Emitter[] emitters;
    size_t[] blockedUntil;    // per cell: explorers skip it until this iteration
    bool stopped;             // stopped before the budget (afterIteration)
    GuideStats guide;         // model-guided explorers
    ScreenStats screen;       // pre-screened improvers
    /// A focused run: from iteration focusFrom on, every emitter improves,
    /// starting and restarting from these cells' elites in turn (refocus);
    /// focusNext counts the starts so far (the next one's turn).
    size_t[] focus;
    size_t focusFrom, focusNext;
    double[] focusScores;     // each focused elite's value at the focus
    bool focusImprovers;      // the focus turned the pre-screened improvers on
}

/// One batch of evaluations as runArchive made it: the seeds (`emitters`
/// empty), or an iteration's, candidate i from emitter i / lambda as it
/// stood when it asked for them.
struct ArchiveBatch(E)
{
    size_t spentBefore;       // evaluations before this batch
    size_t iteration;         // 0 for the seeds
    const(double[])[] genes;  // as evaluated, in [0, 1]
    const(E)[] evals;
    const(Emitter)[] emitters;
    size_t lambda;
}

/// Thrown by runArchive when no seed is within the limits; `hint` (the
/// problem's optional `string noFeasibleSeed() const`) says what to do.
class NoFeasibleSeed : Exception
{
    this(string hint = null, string file = __FILE__, size_t line = __LINE__) pure nothrow @safe
    {
        super("no seed of the archive is within the limits" ~ (hint.length ? ": " ~ hint : ""), file, line);
    }
}

/// An emitter's seed for its restart `restart`.
ulong emitterSeed(ulong seed, size_t k, size_t restart) pure nothrow @nogc @safe
{
    auto r = SplitMix64(seed ^ (0xA4C1_7E00_0000_0000UL + (cast(ulong) k << 20) + restart));
    return r.next();
}

/// What an explorer heading for a target ranks a feasible candidate by
/// (the ranking's value): closer to the target first, and inside it the
/// value.
double explorerGain(double distance, double value) pure nothrow @nogc @safe
{
    return distance > 0 ? -distance : value;
}

/// A model-guided emitter's fitness for a candidate it hasn't evaluated, as
/// its evaluated ones are ranked: an explorer with a target by its distance
/// to it (explorerGain); any other emitter by how far the objective, less
/// `penalty` (the problem's screenPenalty), is above its cell's threshold.
double predictedFitness(G)(const ref G grid, double objective, const Breach breach, const(double)[] descriptors,
                           const(double)[] x, const ref Emitter e, const(double)[] threshold, double penalty,
                           const ref Ranking ranking) pure
    if (isGrid!G)
{
    immutable double net = objective - penalty;
    immutable double gain = e.target != size_t.max ? explorerGain(grid.distance(descriptors, e.target), net)
                                                   : net - threshold[grid.cellOf(descriptors)];
    return ranking.fitness(breach, gain, boundExcess(x));
}

/// Whether an emitter's last RESTART_GENERATIONS generations (`infeasible`:
/// per generation, its candidates over the limits) were nearly all over
/// the limits.
bool nearlyAllInfeasible(const(size_t)[] infeasible, size_t lambda) pure nothrow @nogc @safe
{
    if (infeasible.length < RESTART_GENERATIONS)
        return false;
    size_t recent = 0;
    foreach (v; infeasible[$ - RESTART_GENERATIONS .. $])
        recent += v;
    return recent >= RESTART_INFEASIBLE * RESTART_GENERATIONS * lambda;
}

/// Whether an explorer heading for its target gives it up after a
/// generation, and why: reached (`filled`), stuck, nearly all infeasible,
/// or converged (its step size `sigma`).
RestartReason explorerRestart(const ref Emitter e, bool filled, double sigma, size_t lambda) pure nothrow @nogc @safe
{
    return filled ? RestartReason.reached
         : e.stale >= EXPLORE_STALE ? RestartReason.stuck
         : nearlyAllInfeasible(e.infeasible, lambda) ? RestartReason.infeasible
         : sigma < RESTART_SIGMA ? RestartReason.converged
         : RestartReason.none;
}

/// Whether an improver restarts after a generation, and why: idle, nearly
/// all infeasible, or converged (its step size `sigma`).
RestartReason improverRestart(const ref Emitter e, double sigma, size_t lambda) pure nothrow @nogc @safe
{
    return e.idle >= RESTART_GENERATIONS ? RestartReason.idle
         : nearlyAllInfeasible(e.infeasible, lambda) ? RestartReason.infeasible
         : sigma < RESTART_SIGMA ? RestartReason.converged
         : RestartReason.none;
}

/// Restarts emitter k of a focused run from the next focused cell's elite,
/// at the improvers' step size; pre-screened (the larger population) when
/// the run's improvers are and the model is ready.
void refocus(R)(ref R res, size_t k, bool modelReady) pure nothrow
{
    const opts = &res.options;
    auto e = &res.emitters[k];
    immutable size_t c = res.focus[res.focusNext++ % res.focus.length];
    e.from = c;
    ++e.restarts;
    immutable size_t pop = opts.surrogateImprovers && modelReady ? opts.surrogatePool * res.lambda : res.lambda;
    e.es = CmaEs(res.archive.elites[c].genes, opts.sigma, pop, emitterSeed(opts.seed, k, e.restarts));
    e.idle = 0;
    e.infeasible = null;
}

/// Explorers' targets: the empty cell nearest (in bins) to an occupied
/// one, ties to the lower index, skipping cells given up on lately and
/// other explorers' targets.
size_t pickTarget(R, G)(const ref R res, const ref G grid, size_t k) pure
{
    size_t best = size_t.max;
    long bestD = long.max;
    auto occ = res.archive.occupiedCells;
    foreach (c; 0 .. res.archive.elites.length)
    {
        if (res.archive.elites[c].occupied || res.blockedUntil[c] > res.iterations)
            continue;
        bool taken = false;
        foreach (o, ref other; res.emitters)
            taken |= o != k && other.explorer && other.target == c;
        if (taken)
            continue;
        long d = long.max;
        foreach (o; occ)
            d = grid.binDistance2(c, o) < d ? grid.binDistance2(c, o) : d;
        if (d < bestD)
        {
            best = c;
            bestD = d;
        }
    }
    return best;
}

/// The ranking a run's options give.
private Ranking rankingOf(O)(const ref O opts)
{
    static if (__traits(hasMember, O, "ranking"))
        return opts.ranking;
    else
        return Ranking.init;
}

/// The problem's screenPenalty for `x`, 0 without one.
private double screenPenaltyOf(P)(ref P p, const(double)[] x)
{
    static if (__traits(hasMember, P, "screenPenalty"))
        return p.screenPenalty(x);
    else
        return 0;
}

/// The archive's search over problem `P` and grid `G`, with options `O`
/// (ArchiveOptions or the problem's own, with the same fields), `N` cells
/// fixed at compile time (the archive then held by value; 0: the grid's
/// count, in arrays) and thresholds from `minScore`. A template of
/// functions rather than function templates, so arguments convert as to
/// any function: `Archiving!(P, G).runArchive(p, grid, seeds, options)`.
template Archiving(P, G, O = ArchiveOptions, size_t N = 0, double minScore = 0)
    if (hasDescriptors!P && isGrid!G)
{
    alias Run = ArchiveRun!(P, O, N, minScore);
    alias Generation = ArchiveGeneration!(CausesOf!P);
    alias Batch = ArchiveBatch!(P.Eval);
    alias Model = ArchiveSurrogate!(P.Eval);

    /// Fills an archive from the `seeds` (gene vectors; the feasible ones enter
    /// it), spending opts.evals evaluations on top of the seeds, in whole
    /// iterations. The archive must hold a candidate within the limits before
    /// the emitters start (else NoFeasibleSeed). `N`: the grid's cells when
    /// fixed at compile time (the archive then held by value), else 0.
    ///
    /// With `resume` (a run saved between iterations) it goes on from there in
    /// place of the seeds, with the same options but for opts.evals, which is
    /// still the whole run's: so a run stopped and resumed ends as the run that
    /// never stopped, and a finished run resumed with a larger opts.evals as
    /// the run given that budget at the start. The resumed run is changed in
    /// place, and returned.
    ///
    /// `afterIteration` is called after each iteration, with the run as it
    /// would be saved; returning true stops the run there. `evaluated` gets
    /// every batch of evaluations (the seeds, then each iteration) as soon as
    /// it is evaluated.
    ///
    /// The problem's optional `adoptCauses(ref Causes)` is called on a resumed
    /// run's counts, for what they carry that isn't saved.
    Run runArchive(ref P p, const ref G grid, const(double[])[] seeds, O opts,
                   scope void delegate(const ref Generation) progress = null, Run* resume = null,
                   scope bool delegate(const ref Run) afterIteration = null,
                   scope void delegate(const ref Batch) evaluated = null, Model model = null)
    {
        enforce((model !is null) == (opts.surrogatePool > 0), "model-guided explorers need a model, and only they");
        if (resume !is null)
            return continueArchive(p, grid, opts, *resume, progress, afterIteration, evaluated, model);
        Run res;
        static if (N == 0)
            res.archive = typeof(res.archive)(grid.cells);
        else
            assert(grid.cells == N);
        res.options = opts;
        res.archive.alpha = opts.alpha;
        static if (hasCauses!P)
            res.causes = p.newCauses();
        auto desc = new double[p.descriptorCount];

        // The seeds.
        {
            res.seeds = p.evaluate(seeds, Scoring.feasibleOnly);
            auto genes = new double[][seeds.length];
            foreach (i, x; seeds)
                genes[i] = clamp01(x);
            if (evaluated !is null)
            {
                auto b = Batch(res.evals, 0, genes, res.seeds, null, 0);
                evaluated(b);
            }
            res.evals += seeds.length;
            if (model !is null)
                model.learn(genes, res.seeds);
            foreach (i, ref e; res.seeds)
            {
                static if (hasCauses!P)
                    p.countCauses(res.causes, e);
                immutable Breach b = p.breach(e);
                if (b.feasible)
                {
                    p.descriptors(e, desc);
                    res.archive.offer(grid.cellOf(desc), value(p.objective(e), b), genes[i], e, res.evals);
                }
            }
        }
        if (res.archive.coverage == 0)
        {
            static if (__traits(hasMember, P, "noFeasibleSeed"))
                throw new NoFeasibleSeed(p.noFeasibleSeed);
            else
                throw new NoFeasibleSeed;
        }
        res.seedEvals = res.evals;

        // The emitters, all starting from the best seed.
        immutable size_t first = res.archive.best;
        size_t lambda = opts.lambda;
        if (lambda == 0)
            lambda = CmaEs(res.archive.elites[first].genes, opts.sigma, 0, opts.seed).populationSize;
        res.lambda = lambda;
        res.emitters = new Emitter[opts.emitters];
        immutable size_t explorers = opts.explorers < opts.emitters ? opts.explorers : opts.emitters;
        foreach (k, ref e; res.emitters)
        {
            e.explorer = k >= opts.emitters - explorers;
            e.es = CmaEs(res.archive.elites[first].genes, opts.sigma, lambda, emitterSeed(opts.seed, k, 0));
            e.rng = SplitMix64(opts.seed ^ (0x5E1E_C700UL + k));
        }
        res.blockedUntil = new size_t[res.archive.elites.length];
        foreach (k, ref e; res.emitters)
            if (e.explorer)
                aimExplorer(p, grid, res, k, false, null);
        return continueArchive(p, grid, opts, res, progress, afterIteration, evaluated, model);
    }


    /// Aims explorer k at a new target, restarting it from the elite nearest
    /// the target (with a model and surrogateAim, from where the model puts it
    /// nearest). A model-guided one samples surrogatePool times its population.
    private void aimExplorer(ref P p, const ref G grid, ref Run res, size_t k, bool guided, Model model)
    {
        const opts = &res.options;
        auto e = &res.emitters[k];
        e.target = pickTarget(res, grid, k);
        e.closest = double.infinity;
        e.stale = 0;
        if (e.target == size_t.max)
            return;
        size_t from = size_t.max;
        foreach (c; res.archive.occupiedCells)
            if (from == size_t.max || grid.binDistance2(c, e.target) < grid.binDistance2(from, e.target))
                from = c;
        ++e.restarts;
        immutable size_t pop = guided ? opts.surrogatePool * res.lambda : res.lambda;
        const(double)[] start = res.archive.elites[from].genes;
        if (model !is null && opts.surrogateAim)
        {
            start = aimedStart(p, grid, res, k, model);
            ++res.guide.aims;
        }
        e.es = CmaEs(start, opts.explorerSigma, pop, emitterSeed(opts.seed, k, e.restarts));
        e.idle = 0;
        e.infeasible = null;
    }

    /// Where the model would start explorer k for its target: of every elite
    /// and AIM_SAMPLES points around each (at the explorers' step size, kept
    /// in [0, 1]), the one its fitness ranks best on the model: within the
    /// limits, nearest the target, inside it by value. Ties go to the first,
    /// elites in index order, each before its samples.
    private double[] aimedStart(ref P p, const ref G grid, const ref Run res, size_t k, Model model)
    {
        import std.math : PI, cos, log, sqrt;
        const opts = &res.options;
        const e = &res.emitters[k];
        auto rng = SplitMix64(emitterSeed(opts.seed, k, e.restarts) ^ 0xA1_5EED_A1_5EEDUL);
        double[][] points;
        foreach (c; res.archive.occupiedCells)
        {
            const g = res.archive.elites[c].genes;
            points ~= g.dup;
            foreach (i; 0 .. AIM_SAMPLES)
            {
                auto x = g.dup;
                foreach (ref v; x)
                {
                    // Box-Muller, one draw per gene.
                    immutable double u = 1 - rng.uniform01(), w = rng.uniform01();
                    v += opts.explorerSigma * sqrt(-2 * log(u)) * cos(2 * PI * w);
                    v = clamp01(v);
                }
                points ~= x;
            }
        }
        immutable size_t nd = p.descriptorCount;
        auto objective = new double[points.length], desc = new double[points.length * nd];
        auto breach = new Breach[points.length];
        model.predictArchive(points, objective, breach, desc);
        immutable Ranking ranking = rankingOf(*opts);
        size_t best = 0;
        double bestF = double.infinity;
        foreach (i, x; points)
        {
            immutable double f = predictedFitness(grid, objective[i], breach[i], desc[i * nd .. (i + 1) * nd], x, *e,
                                                  res.archive.threshold[], screenPenaltyOf(p, x), ranking);
            if (f < bestF)
            {
                best = i;
                bestF = f;
            }
        }
        return points[best];
    }


    /// The emitters' iterations of runArchive, from `res` as it stands, until
    /// opts.evals beyond the seeds are spent or `afterIteration` says stop.
    private Run continueArchive(ref P p, const ref G grid, O opts, ref Run res,
                                scope void delegate(const ref Generation) progress,
                                scope bool delegate(const ref Run) afterIteration,
                                scope void delegate(const ref Batch) evaluated, Model model)
    {
        alias E = P.Eval;
        {
            // A resumed run: the same options but for the budget, its emitters,
            // and a model when its emitters sample for one.
            auto was = res.options;
            was.evals = opts.evals;
            enforce(was == opts, "an archive resumed goes on with its own options");
            enforce(res.emitters.length == opts.emitters && res.blockedUntil.length == res.archive.elites.length,
                    "an archive resumed must hold its emitters");
            foreach (ref e; res.emitters)
                enforce(e.es.populationSize == res.lambda
                        || (model !is null && e.es.populationSize == opts.surrogatePool * res.lambda),
                        "this run had model-guided emitters, but no model was given to resume it");
            res.options = opts;
            res.stopped = false;
            static if (__traits(hasMember, P, "adoptCauses"))
                p.adoptCauses(res.causes);
        }
        immutable size_t budget = res.seedEvals + opts.evals;
        immutable size_t lambda = res.lambda;
        immutable Ranking ranking = rankingOf(opts);
        immutable size_t nd = p.descriptorCount;
        auto em = res.emitters;
        auto blockedUntil = res.blockedUntil;
        bool modelReady() { return model !is null && res.iterations >= opts.surrogateFrom; }
        void aim(size_t k) { aimExplorer(p, grid, res, k, modelReady, modelReady ? model : null); }

        auto rows = new double[][](opts.emitters * lambda, p.dimension);
        auto fit = new double[lambda];
        auto improvedAt = new bool[lambda];
        auto desc = new double[nd];
        // Model-guided emitters (their population larger than lambda): every
        // candidate's predicted fitness, and which ones are evaluated.
        auto predicted = new double[][](opts.emitters);
        auto picked = new size_t[][](opts.emitters);
        auto byModel = new size_t[][](opts.emitters);     // the picks the model chose (the rest are random)
        auto predObjective = new double[][](opts.emitters);
        auto predBreach = new Breach[][](opts.emitters);
        auto predDesc = new double[][](opts.emitters);
        while (res.evals < budget)
        {
            ++res.iterations;
            if (modelReady)
            {
                model.train(opts.surrogateEpochs);
                // The model's first iteration: the explorers (and pre-screened
                // improvers) go on from where they are, now sampling the larger
                // population.
                if (res.iterations == opts.surrogateFrom)
                    foreach (k, ref e; em)
                        if ((e.explorer || opts.surrogateImprovers) && e.es.populationSize == lambda)
                        {
                            ++e.restarts;
                            e.es = CmaEs(e.es.currentMean, e.es.stepSize, opts.surrogatePool * lambda,
                                         emitterSeed(opts.seed, k, e.restarts));
                        }
            }
            foreach (k, ref e; em)
            {
                auto pop = e.es.ask();
                if (pop.length == lambda)
                {
                    picked[k] = null;
                    foreach (j, x; pop)
                        rows[k * lambda + j][] = x[];
                    continue;
                }
                // Guided: rank the whole population on the model, evaluate lambda of it.
                assert(modelReady);
                predObjective[k] = new double[pop.length];
                predBreach[k] = new Breach[pop.length];
                predDesc[k] = new double[pop.length * nd];
                model.predictArchive(pop, predObjective[k], predBreach[k], predDesc[k]);
                predicted[k] = new double[pop.length];
                foreach (j, x; pop)
                    predicted[k][j] = predictedFitness(grid, predObjective[k][j], predBreach[k][j],
                                                       predDesc[k][j * nd .. (j + 1) * nd], x, e,
                                                       res.archive.threshold[], screenPenaltyOf(p, x), ranking);
                picked[k] = pickRendered(predicted[k], lambda, opts.surrogateShare, e.rng, byModel[k]);
                foreach (j, i; picked[k])
                    rows[k * lambda + j][] = pop[i][];
            }
            // The problem gets the genes as sampled; the model, the batch and
            // the archive take them clamped, as evaluated.
            auto evals = p.evaluate(rows, Scoring.feasibleOnly);
            auto genes = new double[][rows.length];
            foreach (i; 0 .. rows.length)
                genes[i] = clamp01(rows[i]);
            if (model !is null)
                model.learn(genes, evals);
            immutable size_t spentBefore = res.evals;
            if (evaluated !is null)
            {
                auto b = Batch(spentBefore, res.iterations, genes, evals, em, lambda);
                evaluated(b);
            }
            res.evals += rows.length;

            foreach (k, ref e; em)
            {
                Generation g;
                g.iteration = res.iterations;
                g.emitter = k;
                static if (hasCauses!P)
                    g.causes = p.newCauses();
                size_t infeasible = 0;
                immutable bool exploring = e.explorer && e.target != size_t.max;
                foreach (j; 0 .. lambda)
                {
                    immutable size_t i = k * lambda + j;
                    immutable double excess = boundExcess(rows[i]);
                    const ev = &evals[i];
                    static if (hasCauses!P)
                        p.countCauses(g.causes, *ev);
                    immutable Breach b = p.breach(*ev);
                    improvedAt[j] = false;
                    if (!b.feasible)
                    {
                        ++infeasible;
                        fit[j] = ranking.fitness(b, 0, excess);
                        continue;
                    }
                    p.descriptors(*ev, desc);
                    immutable double v = value(p.objective(*ev), b);
                    immutable Offer o = res.archive.offer(grid.cellOf(desc), v, genes[i], *ev, spentBefore + i + 1);
                    improvedAt[j] = o.improved;
                    g.improved += o.improved;
                    g.newCells += o.newCell;
                    immutable double d = exploring ? grid.distance(desc, e.target) : 0;
                    if (exploring)
                        g.distance = !(g.distance <= d) ? d : g.distance;
                    fit[j] = ranking.fitness(b, exploring ? explorerGain(d, v) : o.value, excess);
                }
                if (picked[k] is null)
                    e.es.tell(fit);
                else
                {
                    // The evaluated candidates with their real fitness, the rest
                    // with the model's.
                    auto all = predicted[k].dup;
                    import std.algorithm : canFind;
                    if (!e.explorer)
                    {
                        res.screen.sampled += all.length;
                        res.screen.random += lambda - byModel[k].length;
                        foreach (j, i; picked[k])
                            if (byModel[k].canFind(i))
                            {
                                res.screen.rendered++;
                                res.screen.predictedFeasible += predBreach[k][i].feasible;
                                res.screen.feasible += p.breach(evals[k * lambda + j]).feasible;
                                res.screen.predictedImproving += predBreach[k][i].feasible && predicted[k][i] < 0;
                                res.screen.improving += improvedAt[j];
                            }
                    }
                    foreach (j, i; picked[k])
                    {
                        all[i] = fit[j];
                        if (!e.explorer || e.target == size_t.max || !byModel[k].canFind(i))
                            continue;
                        const ev = &evals[k * lambda + j];
                        immutable bool inLimits = p.breach(*ev).feasible;
                        res.guide.rendered++;
                        res.guide.predictedFeasible += predBreach[k][i].feasible;
                        res.guide.feasible += inLimits;
                        res.guide.predictedInside += predBreach[k][i].feasible
                                                   && grid.distance(predDesc[k][i * nd .. (i + 1) * nd], e.target) == 0;
                        if (inLimits)
                            p.descriptors(*ev, desc);
                        res.guide.inside += inLimits && grid.cellOf(desc) == e.target;
                    }
                    if (e.target != size_t.max)
                    {
                        res.guide.sampled += all.length;
                        res.guide.random += lambda - byModel[k].length;
                    }
                    e.es.tell(all);
                }
                static if (hasCauses!P)
                    res.causes += g.causes;
                g.explorer = e.explorer;
                g.target = e.target;
                g.focus = res.focus.length ? e.from : size_t.max;
                e.infeasible ~= infeasible;

                g.sigma = e.es.stepSize;
                if (exploring)
                {
                    // Explorer: on to the next target when this one is filled
                    // or given up on.
                    if (g.distance < e.closest - EXPLORE_PROGRESS)
                    {
                        e.closest = g.distance;
                        e.stale = 0;
                    }
                    else
                        ++e.stale;
                    g.restartReason = explorerRestart(e, res.archive.elites[e.target].occupied, g.sigma, lambda);
                    if (g.restartReason != RestartReason.none)
                    {
                        if (g.restartReason == RestartReason.reached)
                            res.reached ~= e.target;
                        else
                        {
                            res.abandoned ~= e.target;
                            blockedUntil[e.target] = res.iterations + EXPLORE_BLOCK;
                        }
                        aim(k);
                    }
                }
                else
                {
                    // An improver (or an explorer without a target, until one
                    // frees up).
                    e.idle = g.improved ? 0 : e.idle + 1;
                    g.restartReason = improverRestart(e, g.sigma, lambda);
                    if (g.restartReason != RestartReason.none)
                    {
                        final switch (g.restartReason)
                        {
                            case RestartReason.idle: ++res.restartsIdle; break;
                            case RestartReason.infeasible: ++res.restartsInfeasible; break;
                            case RestartReason.converged: ++res.restartsConverged; break;
                            case RestartReason.none, RestartReason.reached, RestartReason.stuck: assert(0);
                        }
                        if (res.focus.length)
                            refocus(res, k, modelReady);      // the next focused elite
                        else
                        {
                            auto occ = res.archive.occupiedCells;
                            immutable size_t c = occ[cast(size_t)(e.rng.next() % occ.length)];
                            ++e.restarts;
                            immutable size_t pop = !e.explorer && opts.surrogateImprovers && modelReady
                                                 ? opts.surrogatePool * lambda : lambda;
                            e.es = CmaEs(res.archive.elites[c].genes, opts.sigma, pop,
                                         emitterSeed(opts.seed, k, e.restarts));
                            e.idle = 0;
                            e.infeasible = null;
                        }
                    }
                    if (e.explorer && pickTarget(res, grid, k) != size_t.max)
                        aim(k);
                }
                g.evals = spentBefore + (k + 1) * lambda;
                g.cells = res.archive.coverage;
                g.qdScore = res.archive.qdScore;
                g.bestScore = res.archive.bestScore;
                if (progress !is null)
                    progress(g);
            }
            if (afterIteration !is null && res.evals < budget && afterIteration(res))
            {
                res.stopped = true;
                break;
            }
        }
        return res;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

version (unittest)
{
    /// Two genes; its descriptors are the genes, its objective prefers the
    /// middle, and the corner beyond a + b = 1.4 is over its limit.
    private struct Plane
    {
        alias Eval = Outcome;
        alias Causes = Count;
        struct Count
        {
            size_t total;
            void opOpAssign(string op : "+")(const Count o) { total += o.total; }
        }
        size_t calls;

        size_t dimension() const { return 2; }
        Outcome[] evaluate(const(double[])[] genes, Scoring)
        {
            ++calls;
            auto o = new Outcome[genes.length];
            foreach (i, x; genes)
            {
                immutable double a = clamp01(x[0]), b = clamp01(x[1]);
                o[i].scored = true;
                o[i].objective = 10 - (a - 0.5) ^^ 2 - (b - 0.5) ^^ 2;
                immutable double over = a + b - 1.4;
                o[i].breach = Breach(over > 0 ? over : 0);
                o[i].descriptors = [a, b];
            }
            return o;
        }
        bool scored(const ref Outcome e) const { return e.scored; }
        double objective(const ref Outcome e) const { return e.objective; }
        Breach breach(const ref Outcome e) const { return e.breach; }
        size_t descriptorCount() const { return 2; }
        void descriptors(const ref Outcome e, double[] d) const { d[] = e.descriptors[]; }
        Count newCauses() const { return Count(0); }
        void countCauses(ref Count c, const ref Outcome e) const { c.total += !e.breach.feasible; }
    }

    /// A model that knows the problem exactly.
    private final class Oracle : ArchiveSurrogate!Outcome
    {
        size_t learned, trained;
        void learn(const(double[])[] genes, const(Outcome)[] evals) { learned += genes.length; }
        void train(size_t epochs) { trained += epochs; }
        void predictArchive(const(double[])[] genes, double[] objective, Breach[] breach, double[] descriptors)
        {
            Plane p;
            foreach (i, ref e; p.evaluate(genes, Scoring.all))
            {
                objective[i] = e.objective;
                breach[i] = e.breach;
                descriptors[2 * i .. 2 * i + 2] = e.descriptors[];
            }
        }
    }

    private BinnedGrid planeGrid()
    {
        return BinnedGrid([Axis("a", [0.25, 0.5, 0.75], "abcd", 0, 0.25), Axis("b", [0.25, 0.5, 0.75], "abcd", 0, 0.25)]);
    }

    private ArchiveOptions planeOptions()
    {
        ArchiveOptions o;
        o.evals = 400;
        o.emitters = 3;
        o.explorers = 1;
        o.lambda = 8;
        o.seed = 5;
        return o;
    }

    /// Bit for bit, NaN included (a CMA-ES state holds one before its
    /// first spare draw, and NaN == NaN is false).
    private bool sameBits(T)(const T a, const T b)
    {
        static if (is(T : const(double)))
            return a is b;
        else static if (is(T : const(U)[], U))
        {
            if (a.length != b.length)
                return false;
            foreach (i; 0 .. a.length)
                if (!sameBits!U(a[i], b[i]))
                    return false;
            return true;
        }
        else static if (is(T == struct))
        {
            foreach (i, ref v; a.tupleof)
                if (!sameBits(v, b.tupleof[i]))
                    return false;
            return true;
        }
        else
            return a == b;
    }

    private bool sameRun(R)(const ref R a, const ref R b)
    {
        if (a.evals != b.evals || a.iterations != b.iterations || a.archive.threshold != b.archive.threshold
            || a.reached != b.reached || a.abandoned != b.abandoned || a.causes.total != b.causes.total)
            return false;
        foreach (c, ref el; a.archive.elites)
            if (el.occupied != b.archive.elites[c].occupied || el.genes != b.archive.elites[c].genes
                || el.found != b.archive.elites[c].found || el.value != b.archive.elites[c].value)
                return false;
        foreach (k, ref e; a.emitters)
            if (!sameBits(e.es.save(), b.emitters[k].es.save()) || e.rng != b.emitters[k].rng)
                return false;
        return true;
    }
}

unittest
{
    // CMA-MAE thresholds and the result archive, by value and by array.
    static foreach (N; [0, 4])
    {{
        static if (N)
            Archive!(Outcome, N) a;
        else
            auto a = Archive!Outcome(4);
        a.alpha = 0.5;
        Outcome e;
        assert(a.offer(1, 20, [0.1], e, 1) == Offer(20, true, true));
        assert(a.threshold[1] == 10 && a.elites[1].value == 20);
        // Below the elite but above the threshold: improves and moves the
        // threshold, but the elite stays.
        assert(a.offer(1, 15, [0.2], e, 2) == Offer(5, true, false));
        assert(a.threshold[1] == 12.5 && a.elites[1].genes == [0.1]);
        // Below the threshold: negative value, nothing changes.
        assert(a.offer(1, 12, [0.3], e, 3) == Offer(-0.5, false, false));
        assert(a.threshold[1] == 12.5);
        a.offer(1, 21, [0.4], e, 4);
        assert(a.elites[1].genes == [0.4] && a.elites[1].found == 4);
        a.offer(3, 2, [0.5], e, 5);
        assert(a.coverage == 2 && a.best == 1 && a.qdScore == 23 && a.bestScore == 21 && a.occupiedCells == [1, 3]);
    }}
}

unittest
{
    // The diverse selection: the best first, then the farthest within the
    // window; ties to the lower index.
    const grid = planeGrid();
    auto a = Archive!Outcome(grid.cells);
    Outcome e;
    a.offer(grid.parse("dd"), 26, [0.5], e, 1);
    a.offer(grid.parse("dc"), 25.9, [0.5], e, 1);
    a.offer(grid.parse("da"), 25.5, [0.5], e, 1);
    a.offer(grid.parse("aa"), 23.9, [0.5], e, 1);       // farthest, but more than 2 below
    assert(selectDiverse(a, grid, 2, 9) == [grid.parse("dd"), grid.parse("da"), grid.parse("dc")]);
    assert(selectDiverse(a, grid, 1, 2) == [grid.parse("dd"), grid.parse("da")]);
}

unittest
{
    // Restart decisions.
    immutable size_t lambda = 10;
    assert(!nearlyAllInfeasible([10, 10, 10, 10], lambda));
    assert(nearlyAllInfeasible([0, 9, 9, 9, 9, 9], lambda));
    assert(!nearlyAllInfeasible([9, 9, 9, 9, 8], lambda));
    Emitter e;
    assert(explorerRestart(e, false, 0.1, lambda) == RestartReason.none);
    assert(explorerRestart(e, false, RESTART_SIGMA / 2, lambda) == RestartReason.converged);
    e.infeasible = [10, 10, 10, 10, 10];
    assert(explorerRestart(e, false, RESTART_SIGMA / 2, lambda) == RestartReason.infeasible);
    e.stale = EXPLORE_STALE;
    assert(explorerRestart(e, false, 0.1, lambda) == RestartReason.stuck);
    assert(explorerRestart(e, true, 0.1, lambda) == RestartReason.reached);
    Emitter m;
    assert(improverRestart(m, 0.1, lambda) == RestartReason.none);
    m.infeasible = [10, 10, 10, 10, 10];
    assert(improverRestart(m, RESTART_SIGMA / 2, lambda) == RestartReason.infeasible);
    m.idle = RESTART_GENERATIONS;
    assert(improverRestart(m, 0.1, lambda) == RestartReason.idle);
    assert(reasonText(RestartReason.none) == "" && reasonText(RestartReason.stuck) == "stuck");

    // A prediction ranks as the candidate it predicts: an improver by its
    // value over the threshold, an explorer by its distance to the target.
    const grid = planeGrid();
    auto th = new double[grid.cells];
    th[] = 0;
    th[grid.parse("bb")] = 9;
    Emitter improver, explorer;
    explorer.target = grid.parse("cc");
    immutable Ranking r;
    assert(predictedFitness(grid, 10, Breach(0), [0.3, 0.3], [0.5], improver, th, 0.5, r) == -0.5);
    immutable double d = grid.distance([0.3, 0.3], explorer.target);
    assert(d > 0 && predictedFitness(grid, 10, Breach(0), [0.3, 0.3], [0.5], explorer, th, 0, r) == d);
    assert(predictedFitness(grid, 10, Breach(0.5), [0.3, 0.3], [0.5], improver, th, 0, r) == r.hardOffset + 0.5);
}

unittest
{
    // A run fills cells, the same on every call; stopped and resumed it
    // ends as the run that never stopped; with an exact model the
    // explorers are guided, and that too resumes exactly.
    alias A = Archiving!(Plane, BinnedGrid);
    const grid = planeGrid();
    auto o = planeOptions();
    alias Run = ArchiveRun!Plane;
    Plane p;
    auto whole = A.runArchive(p, grid, [[0.5, 0.5]], o);
    assert(whole.evals >= 400 && whole.archive.coverage > 4 && whole.causes.total > 0);
    assert(whole.archive.elites[whole.archive.best].genes.length == 2);
    Plane q;
    auto again = A.runArchive(q, grid, [[0.5, 0.5]], o);
    assert(sameRun(whole, again));

    size_t generations;
    Plane s;
    auto stopped = A.runArchive(s, grid, [[0.5, 0.5]], o, (ref const ArchiveGeneration!(Plane.Count) g) {
        ++generations;
        assert(g.emitter < 3 && g.evals > 0);
    }, null, (ref const Run r) => r.iterations == 4);
    assert(stopped.stopped && stopped.iterations == 4 && generations == 4 * 3);
    Plane t;
    auto resumed = A.runArchive(t, grid, null, o, null, &stopped);
    assert(!resumed.stopped && sameRun(whole, resumed));

    // A larger budget given to the finished run: as the run given it at the start.
    o.evals = 600;
    Plane u, v;
    auto longer = A.runArchive(u, grid, [[0.5, 0.5]], o);
    auto extended = A.runArchive(v, grid, null, o, null, &whole);
    assert(sameRun(longer, extended));

    // Guided.
    auto g = planeOptions();
    g.surrogatePool = 4;
    g.surrogateFrom = 2;
    g.surrogateImprovers = true;
    g.surrogateAim = true;
    Plane w, x;
    auto m1 = new Oracle, m2 = new Oracle;
    size_t batches;
    auto guided = A.runArchive(w, grid, [[0.5, 0.5]], g, null, null, null, (ref const ArchiveBatch!Outcome b) {
        assert(b.genes.length == b.evals.length && (b.iteration == 0) == (b.emitters.length == 0));
        ++batches;
    }, m1);
    assert(guided.guide.sampled > 0 && guided.screen.sampled > 0 && m1.trained > 0 && m1.learned == guided.evals);
    assert(batches == guided.iterations + 1);
    auto gs = A.runArchive(x, grid, [[0.5, 0.5]], g, null, null, (ref const Run r) => r.iterations == 3, null, m2);
    Plane y;
    auto gr = A.runArchive(y, grid, null, g, null, &gs, null, null, m2);
    assert(sameRun(guided, gr));
    import std.exception : assertThrown;
    Plane z;
    assertThrown(A.runArchive(z, grid, [[0.5, 0.5]], g));        // a model-guided run needs its model
}

unittest
{
    // A focused run restarts its emitters from the focused elites in turn;
    // no seed within the limits throws NoFeasibleSeed.
    alias A = Archiving!(Plane, BinnedGrid);
    const grid = planeGrid();
    auto o = planeOptions();
    Plane p;
    auto r = A.runArchive(p, grid, [[0.5, 0.5]], o, null, null, (ref const A.Run r) => r.iterations == 2);
    auto cells = r.archive.occupiedCells;
    r.focus = cells[0 .. 2].dup;
    r.focusFrom = r.iterations;
    foreach (k; 0 .. r.emitters.length)
        refocus(r, k, false);
    assert(r.focusNext == 3 && r.emitters[0].from == cells[0] && r.emitters[1].from == cells[1]
           && r.emitters[2].from == cells[0] && r.emitters[0].es.currentMean == r.archive.elites[cells[0]].genes);
    Plane q;
    auto f = A.runArchive(q, grid, null, o, null, &r);
    assert(f.focus.length == 2 && f.evals >= 400);

    Plane s;
    import std.exception : assertThrown;
    assertThrown!NoFeasibleSeed(A.runArchive(s, grid, [[1.0, 1.0]], o));
}
