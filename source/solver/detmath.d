/// log2 and exp2 for the per-sample audio path, written out in plain IEEE
/// arithmetic so they give the same bits on every platform. The system
/// maths library (glibc, the MSVC runtime) is free to differ in the last
/// bit, which is harmless for a filter design but not for a function that
/// runs on every sample of a scored mix.
///
/// Each comes as a scalar and as a 4-lane vector (one AVX register). Both do
/// exactly the same operations, so every lane equals the scalar result bit
/// for bit, and a loop can run vectors with a scalar tail. Accuracy is about
/// 1e-15 relative, far below anything audible.
module solver.detmath;

import std.traits : Unqual;

import ldc.simd : greaterMask;

alias V  = __vector(double[4]);
alias VL = __vector(long[4]);

private enum double SQRT2 = 1.4142135623730951;
private enum double INV_LN2 = 1.4426950408889634;
private enum double LN2 = 0.6931471805599453;
private enum double TWO52 = 4503599627370496.0;          // 2^52
private enum double ROUND = 6755399441055744.0;          // 1.5 · 2^52: adding it rounds to an integer
private enum long   MANTISSA = 0x000F_FFFF_FFFF_FFFFL;
private enum long   ONE = 0x3FF0_0000_0000_0000L;        // the bits of 1.0
private enum long   TWO52_BITS = 0x4330_0000_0000_0000L; // the bits of 2^52

/// log2(x) for finite x > 0. Denormals and zero are not handled: callers
/// floor the input first.
pragma(inline, true)
double log2d(double x) pure nothrow @nogc @trusted
{
    // x = m · 2^e. The exponent field is turned into a double exactly by
    // planting it in the mantissa of 2^52 (no int64 conversion: AVX2 has no
    // vector form of it).
    immutable long bits = *cast(long*) &x;
    long eb = ((bits >>> 52) & 0x7FF) | TWO52_BITS;
    long mb = (bits & MANTISSA) | ONE;
    double e = (*cast(double*) &eb - TWO52) - 1023;
    double m = *cast(double*) &mb;                      // in [1, 2)
    immutable bool hi = m > SQRT2;
    m = hi ? m * 0.5 : m;
    e = hi ? e + 1 : e;
    return e + logPoly(m);
}

/// ditto, per lane.
pragma(inline, true)
V log2v(V x) pure nothrow @nogc
{
    immutable VL bits = cast(VL) x;
    VL eb = ((bits >>> 52) & 0x7FF) | TWO52_BITS;
    VL mb = (bits & MANTISSA) | ONE;
    V e = (cast(V) eb - TWO52) - 1023;
    V m = cast(V) mb;
    immutable VL hi = greaterMask!V(m, SQRT2);
    m = select(hi, m * 0.5, m);
    e = select(hi, e + 1, e);
    return e + logPoly(m);
}

/// 2^y for y in [-1000, 1000]; exactly 1 at y = 0.
pragma(inline, true)
double exp2d(double y) pure nothrow @nogc @trusted
{
    // y = n + f with n the nearest integer, f in [-1/2, 1/2].
    immutable double t = y + ROUND;
    immutable double f = y - (t - ROUND);
    long sb = (*cast(long*) &t + 1023) << 52;           // the bits of 2^n
    return expPoly(f * LN2) * *cast(double*) &sb;
}

/// ditto, per lane.
pragma(inline, true)
V exp2v(V y) pure nothrow @nogc
{
    immutable V t = y + ROUND;
    immutable V f = y - (t - ROUND);
    immutable VL sb = (cast(VL) t + 1023) << 52;
    return expPoly(f * LN2) * cast(V) sb;
}

/// Lane-wise `mask ? a : b` for a comparison mask.
pragma(inline, true)
V select(VL mask, V a, V b) pure nothrow @nogc
{
    return cast(V)((mask & cast(VL) a) | (~mask & cast(VL) b));
}

private:

/// log2 m for m in [sqrt(1/2), sqrt(2)): 2 atanh(s) / ln 2, s = (m-1)/(m+1).
pragma(inline, true)
Unqual!T logPoly(T)(T m)
{
    alias U = Unqual!T;
    immutable U s = (m - 1) / (m + 1);
    immutable U s2 = s * s;
    U p = s2 * (1.0 / 19) + 1.0 / 17;
    p = p * s2 + 1.0 / 15;
    p = p * s2 + 1.0 / 13;
    p = p * s2 + 1.0 / 11;
    p = p * s2 + 1.0 / 9;
    p = p * s2 + 1.0 / 7;
    p = p * s2 + 1.0 / 5;
    p = p * s2 + 1.0 / 3;
    p = p * s2 + 1;
    return (s * 2) * p * INV_LN2;
}

/// e^z for |z| <= ln(2)/2, Taylor to z^13 (error below 1e-17); exactly 1
/// at z = 0.
pragma(inline, true)
Unqual!T expPoly(T)(T z)
{
    alias U = Unqual!T;
    U p = z * (1.0 / 6_227_020_800) + 1.0 / 479_001_600;
    p = p * z + 1.0 / 39_916_800;
    p = p * z + 1.0 / 3_628_800;
    p = p * z + 1.0 / 362_880;
    p = p * z + 1.0 / 40_320;
    p = p * z + 1.0 / 5_040;
    p = p * z + 1.0 / 720;
    p = p * z + 1.0 / 120;
    p = p * z + 1.0 / 24;
    p = p * z + 1.0 / 6;
    p = p * z + 0.5;
    p = p * z + 1;
    return p * z + 1;
}

unittest
{
    import std.math : exp2, log2, abs;

    foreach (i; 0 .. 20_000)
    {
        immutable double x = 1e-12 * exp2(i * 0.0025);             // 1e-12 .. ~1e3
        immutable double a = log2d(x), b = log2(x);
        assert(abs(a - b) <= 1e-13 * (abs(b) > 1 ? abs(b) : 1));
    }
    foreach (i; 0 .. 20_000)
    {
        immutable double y = -60 + i * 0.0037;                      // -60 .. 14
        immutable double a = exp2d(y), b = exp2(y);
        assert(abs(a - b) <= 2e-15 * b);
    }
    assert(exp2d(0) == 1);
    assert(exp2d(-3) == 0.125);
    assert(log2d(1) == 0);
    assert(log2d(0.125) == -3);

    // Every vector lane equals the scalar result bit for bit.
    foreach (i; 0 .. 5_000)
    {
        V x, y;
        foreach (l; 0 .. 4)
        {
            x.array[l] = 1e-9 * exp2(i * 0.007 + l * 1.37);
            y.array[l] = -50 + i * 0.011 + l * 0.29;
        }
        immutable V lx = log2v(x), ey = exp2v(y);
        foreach (l; 0 .. 4)
        {
            assert(lx.array[l] == log2d(x.array[l]));
            assert(ey.array[l] == exp2d(y.array[l]));
        }
    }
}

// Reference bit patterns, so a platform (or a compiler, or flags) whose
// arithmetic differs from the one these were taken on fails here. Taken on
// Linux x86-64 with LDC 1.42.0; the same bits with and without FMA and
// -mcpu=native. Pairs of input and result bits.
unittest
{
    static immutable ulong[2][] logRef = [
        [0x01A56E1FC2F8F359, 0xC08F24A09F1A8B89], // log2d(1e-300)
        [0x3D719799812DEA11, 0xC043EE7B471B3A95], // log2d(9.9999999999999998e-13)
        [0x3E9421F5F40D8376, 0xC035AB250E10732D], // log2d(2.9999999999999999e-07)
        [0x3F50624DD2F1A9FC, 0xC023EE7B471B3A95], // log2d(0.001)
        [0x3FB999999999999A, 0xC00A934F0979A371], // log2d(0.10000000000000001)
        [0x3FE0000000000000, 0xBFF0000000000000], // log2d(0.5)
        [0x3FE6A09E667F3BCD, 0xBFDFFFFFFFFFFFFE], // log2d(0.70710678118654757)
        [0x3FE8000000000000, 0xBFDA8FF971810A5D], // log2d(0.75)
        [0x3FF0000000000005, 0x3CDCDA993E7663BB], // log2d(1.0000000000000011)
        [0x3FF6A09E667F3BCD, 0x3FE0000000000001], // log2d(1.4142135623730951)
        [0x3FF6A09E667F3BCE, 0x3FE0000000000003], // log2d(1.4142135623730954)
        [0x4004000000000000, 0x3FF5269E12F346E2], // log2d(2.5)
        [0x4008000000000000, 0x3FF95C01A39FBD69], // log2d(3)
        [0x4024000000000000, 0x400A934F0979A371], // log2d(10)
        [0x407B800000000000, 0x4021900E6160002D], // log2d(440)
        [0x408F400000000000, 0x4023EE7B471B3A95], // log2d(1000)
        [0x40E7700000000000, 0x402F19FB7B8F3242], // log2d(48000)
        [0x426D1A94A2000000, 0x4043EE7B471B3A95], // log2d(1000000000000)
        [0x7E37E43C8800759C, 0x408F24A09F1A8B89], // log2d(1.0000000000000001e+300)
    ];
    static immutable ulong[2][] expRef = [
        [0xC08F400000000000, 0x0170000000000000], // exp2d(-1000)
        [0xC04E000000000000, 0x3C30000000000000], // exp2d(-60)
        [0xC02A934F0979A371, 0x3F1A36E2EB1C4330], // exp2d(-13.287712379549449)
        [0xBFF8000000000000, 0x3FD6A09E667F3BCC], // exp2d(-1.5)
        [0xBFE0000000000000, 0x3FE6A09E667F3BCC], // exp2d(-0.5)
        [0xBE112E0BE826D695, 0x3FEFFFFFFFA0BC0D], // exp2d(-1.0000000000000001e-09)
        [0x3C670EF54646D497, 0x3FF0000000000000], // exp2d(1.0000000000000001e-17)
        [0x3FD3333333333333, 0x3FF3B2C47BFF8329], // exp2d(0.29999999999999999)
        [0x3FE0000000000000, 0x3FF6A09E667F3BCC], // exp2d(0.5)
        [0x3FDFFFFFFFFFFFFF, 0x3FF6A09E667F3BCC], // exp2d(0.49999999999999994)
        [0x3FB5555555555555, 0x3FF0F38F92D97963], // exp2d(0.083333333333333329)
        [0x3FE2AAAAAAAAAAAB, 0x3FF7F910D768CFB0], // exp2d(0.58333333333333337)
        [0x400A934F0979A371, 0x4023FFFFFFFFFFFF], // exp2d(3.3219280948873622)
        [0x4024000000000000, 0x4090000000000000], // exp2d(10)
        [0x402B000000000000, 0x40C6A09E667F3BCC], // exp2d(13.5)
        [0x404FE00000000000, 0x43EAE89F995AD3AE], // exp2d(63.75)
        [0x408F400000000000, 0x7E70000000000000], // exp2d(1000)
    ];

    static double d(ulong b) pure nothrow @nogc @trusted { return *cast(double*) &b; }
    static ulong u(double x) pure nothrow @nogc @trusted { return *cast(ulong*) &x; }

    foreach (r; logRef)
        assert(u(log2d(d(r[0]))) == r[1]);
    foreach (r; expRef)
        assert(u(exp2d(d(r[0]))) == r[1]);

    // And through the vectors, four references at a time.
    foreach (i; 0 .. expRef.length - 3)   // the shorter list
    {
        V x, y;
        foreach (l; 0 .. 4)
        {
            x.array[l] = d(logRef[i + l][0]);
            y.array[l] = d(expRef[i + l][0]);
        }
        immutable V lx = log2v(x), ey = exp2v(y);
        foreach (l; 0 .. 4)
        {
            assert(u(lx.array[l]) == logRef[i + l][1]);
            assert(u(ey.array[l]) == expRef[i + l][1]);
        }
    }
}
