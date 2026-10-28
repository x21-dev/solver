/// A small counter-free PRNG with a fixed algorithm, so seeded noise is
/// bit-identical on every platform and compiler (std.random's defaults are
/// not guaranteed to stay the same across Phobos versions).
module solver.rng;

/// SplitMix64 (Steele, Lea, Flood 2014).
struct SplitMix64
{
pure nothrow @nogc @safe:

    ulong state;

    this(ulong seed) { state = seed; }

    ulong next()
    {
        ulong z = (state += 0x9E3779B97F4A7C15UL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// Uniform from 0 up to, not including, 1; 53 bits.
    double uniform01() { return (next() >> 11) * (1.0 / (1UL << 53)); }

    /// Uniform from -1 up to, not including, 1.
    double bipolar() { return 2.0 * uniform01() - 1.0; }
}

unittest
{
    // Reference values for seed 0, from the published algorithm.
    auto r = SplitMix64(0);
    assert(r.next() == 0xE220A8397B1DCDAFUL);
    assert(r.next() == 0x6E789E6AA1B965F4UL);
}
