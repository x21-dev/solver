/// Little-endian bytes of values and arrays, for saved states: the
/// ensemble's and, in pleaser, the archive model's and checkpoints'.
module solver.bytes;

/// Little-endian bytes of values and arrays (an array with its length
/// first unless given), for the saved states of the ensemble and the archive's model.
struct Bytes
{
    ubyte[] data;
    size_t at;

    void put(T)(T v) if (!is(T : const(U)[], U))
    {
        static if (is(T == bool))
            data ~= cast(ubyte)(v ? 1 : 0);
        else
            data ~= (cast(const(ubyte)*) &v)[0 .. T.sizeof];
    }
    void put(T)(const(T)[] a)
    {
        put(a.length);
        foreach (v; a)
            put(cast() v);
    }
    T get(T)()
    {
        import std.exception : enforce;
        static if (is(T == bool))
        {
            enforce(at < data.length, "saved state cut short");
            return data[at++] != 0;
        }
        else
        {
            enforce(at + T.sizeof <= data.length, "saved state cut short");
            T v;
            (cast(ubyte*) &v)[0 .. T.sizeof] = data[at .. at + T.sizeof];
            at += T.sizeof;
            return v;
        }
    }
    T[] getArray(T)(size_t expect = size_t.max)
    {
        import std.exception : enforce;
        immutable size_t n = get!size_t;
        enforce(expect == size_t.max || n == expect, "saved state of another shape");
        enforce(n <= (data.length - at) / T.sizeof, "saved state cut short");
        auto o = new T[n];
        foreach (ref v; o)
            v = get!T;
        return o;
    }
    bool done() const pure nothrow @nogc { return at == data.length; }
}

unittest
{
    // Values and arrays come back as they went in; a short read or an array
    // of another length throws.
    import std.exception : assertThrown;
    Bytes o;
    o.put(3.5);
    o.put(true);
    o.put(cast(size_t) 7);
    o.put([1.0, -2.0]);
    auto r = Bytes(o.data.dup);
    assert(r.get!double == 3.5 && r.get!bool && r.get!size_t == 7);
    assert(r.getArray!double(2) == [1.0, -2.0] && r.done);
    assertThrown(Bytes(o.data.dup[0 .. 4]).get!double);
    auto s = Bytes(o.data.dup);
    s.at = 8 + 1 + 8;
    assertThrown(s.getArray!double(3));
}
