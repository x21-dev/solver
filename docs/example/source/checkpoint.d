/// A checkpoint file for the example: solver gives a run's state as plain
/// data (RunState, the archive, the causes) and the model's as bytes, and
/// leaves the file format to the caller. This one writes every field in
/// declaration order with solver.bytes, little-endian, arrays with their
/// length first. A field added to any of these types changes the format,
/// so a real project should version its files.
module checkpoint;

import std.exception : enforce;

import solver.bytes : Bytes;

/// Appends `v` to `o`: a struct field by field, a static array element by
/// element, a slice its length then its elements, a scalar its bytes.
void write(T)(ref Bytes o, const ref T v)
{
    static if (is(T == struct))
    {
        foreach (ref f; v.tupleof)
            write(o, f);
    }
    else static if (is(T == E[n], E, size_t n))
    {
        foreach (ref e; v)
            write(o, e);
    }
    else static if (is(T == U[], U))
    {
        o.put(v.length);
        foreach (ref e; v)
            write(o, e);
    }
    else
        o.put(cast() v);
}

/// Reads into `v` what `write` wrote.
void read(T)(ref Bytes r, ref T v)
{
    static if (is(T == struct))
    {
        foreach (ref f; v.tupleof)
            read(r, f);
    }
    else static if (is(T == E[n], E, size_t n))
    {
        foreach (ref e; v)
            read(r, e);
    }
    else static if (is(T == immutable(char)[]))
    {
        char[] s;
        read(r, s);
        v = s.idup;
    }
    else static if (is(T == U[], U))
    {
        immutable size_t n = r.get!size_t;
        enforce(n <= r.data.length - r.at, "checkpoint cut short");
        v = new U[n];
        foreach (ref e; v)
            read(r, e);
    }
    else
        v = r.get!T;
}

unittest
{
    static struct S { double a; int[2] b; string s; double[][] m; bool f; }
    const S x = {1.5, [3, 4], "beam", [[1.0], [], [2.0, 3.0]], true};
    Bytes o;
    write(o, x);
    auto r = Bytes(o.data);
    S y;
    read(r, y);
    assert(y == x && r.done);
}
