/// A checkpoint file format to copy into a project and adapt (the developer
/// manual's "Stopping, saving and resuming → To a file"). solver gives a
/// run's state as plain data (RunState, the archive, the causes) and the
/// model's as bytes, and leaves the file format to the caller.
///
/// A file is a header (the project's magic string and its format version)
/// and one value. A struct is written as its named fields, each with its
/// byte length, so the reader matches fields by name: a field the file
/// lacks keeps the default the reader starts from (older files stay
/// readable when a field is added), and a field the reader doesn't know is
/// refused (a newer file isn't half read) unless it is listed as retired.
/// A static array is its elements, a slice its length then its elements, a
/// scalar its bytes; lengths are ulong. Everything goes through
/// solver.bytes, so the bytes are little-endian on little-endian machines.
module checkpoint;

import std.exception : enforce;
import std.traits : isPointer, isAssociativeArray, isDelegate, isFunctionPointer;

import solver.bytes : Bytes;

/// Appends the header: `magic`, then `version_`.
void writeHeader(ref Bytes o, string magic, uint version_)
{
    write(o, magic);
    o.put(version_);
}

/// Reads the header `writeHeader` wrote and returns its version. Throws
/// when the magic isn't `magic` (not this project's file) or the version
/// is above `current` (a newer program's file).
uint readHeader(ref Bytes r, string magic, uint current)
{
    string m;
    read(r, m);
    enforce(m == magic, "not a " ~ magic ~ " file");
    immutable uint v = r.get!uint;
    enforce(v <= current, "the file is of a newer format than this program reads");
    return v;
}

/// Appends `v` to `o`: a struct its fields by name, a static array element
/// by element, a slice its length then its elements, a scalar its bytes.
void write(T)(ref Bytes o, const ref T v)
{
    static assert(!isPointer!T && !isAssociativeArray!T && !isDelegate!T && !isFunctionPointer!T
                  && !is(T == class) && !is(T == interface) && !is(T == union),
                  T.stringof ~ " holds a reference; save what it refers to instead");
    static if (is(T == struct))
    {
        o.put(ulong(v.tupleof.length));
        static foreach (i; 0 .. v.tupleof.length)
        {{
            immutable string name = __traits(identifier, T.tupleof[i]);
            write(o, name);
            // The field's byte length, filled in once it is written.
            immutable size_t at = o.data.length;
            o.put(ulong(0));
            write(o, v.tupleof[i]);
            immutable ulong n = o.data.length - at - ulong.sizeof;
            o.data[at .. at + ulong.sizeof] = (cast(const(ubyte)*) &n)[0 .. ulong.sizeof];
        }}
    }
    else static if (is(T == E[n], E, size_t n))
    {
        o.put(ulong(n));
        foreach (ref e; v)
            write(o, e);
    }
    else static if (is(T == U[], U))
    {
        o.put(ulong(v.length));
        foreach (ref e; v)
            write(o, e);
    }
    else
        o.put(cast() v);
}

/// Reads into `v` what `write` wrote. A struct field the file lacks keeps
/// the value `v` has, so start from a default (`T v;`). A field in the file
/// that `T` has no longer is skipped when `retired` names it as
/// "Struct.field" (`Struct` the type's name without template arguments),
/// and throws otherwise.
void read(T)(ref Bytes r, ref T v, const(string)[] retired = null)
{
    static if (is(T == struct))
    {
        immutable ulong fields = r.get!ulong;
        foreach (_; 0 .. fields)
        {
            string name;
            read(r, name);
            immutable ulong n = r.get!ulong;
            enforce(n <= r.data.length - r.at, "checkpoint cut short");
            auto field = Bytes(r.data[r.at .. r.at + cast(size_t) n]);
            r.at += cast(size_t) n;
            bool known;
            static foreach (i; 0 .. v.tupleof.length)
                if (name == __traits(identifier, T.tupleof[i]))
                {
                    read(field, v.tupleof[i], retired);
                    enforce(field.done, "checkpoint field " ~ name ~ " of another type");
                    known = true;
                }
            if (!known)
            {
                import std.algorithm : canFind;
                enforce(retired.canFind(__traits(identifier, T) ~ "." ~ name),
                        "checkpoint field " ~ __traits(identifier, T) ~ "." ~ name ~ " unknown to this program");
            }
        }
    }
    else static if (is(T == E[n], E, size_t n))
    {
        enforce(r.get!ulong == n, "checkpoint array of another length");
        foreach (ref e; v)
            read(r, e, retired);
    }
    else static if (is(T == immutable(char)[]))
    {
        char[] s;
        read(r, s);
        v = s.idup;
    }
    else static if (is(T == U[], U))
    {
        immutable ulong n = r.get!ulong;
        enforce(n <= r.data.length - r.at, "checkpoint cut short");
        v = new U[cast(size_t) n];
        foreach (ref e; v)
            read(r, e, retired);
    }
    else
        v = r.get!T;
}

unittest
{
    import std.exception : assertThrown;

    // Values come back as they went in, whatever they nest.
    static struct S { double a; int[2] b; string s; double[][] m; bool f; }
    const S x = {1.5, [3, 4], "beam", [[1.0], [], [2.0, 3.0]], true};
    Bytes o;
    writeHeader(o, "test", 1);
    write(o, x);
    auto r = Bytes(o.data);
    assert(readHeader(r, "test", 1) == 1);
    S y;
    read(r, y);
    assert(y == x && r.done);

    // Another project's file, or a newer format's, is refused.
    auto other = Bytes(o.data);
    assertThrown(readHeader(other, "other", 1));
    Bytes newer;
    writeHeader(newer, "test", 2);
    auto rn = Bytes(newer.data);
    assertThrown(readHeader(rn, "test", 1));

    // A field added since: the older file still reads, the new field keeps
    // its default. Fields are matched by name, so their order may change.
    static struct Added { bool f; string s; double a; int[2] b; double[][] m; int count = 7; }
    Added z;
    auto ra = Bytes(o.data);
    readHeader(ra, "test", 1);
    read(ra, z);
    assert(z.a == 1.5 && z.b == [3, 4] && z.s == "beam" && z.m == x.m && z.f && z.count == 7 && ra.done);

    // A field dropped since: refused unless retired.
    static struct Dropped { double a; int[2] b; string s; bool f; }
    Dropped d;
    auto rd = Bytes(o.data);
    readHeader(rd, "test", 1);
    assertThrown(read(rd, d));
    rd = Bytes(o.data);
    readHeader(rd, "test", 1);
    read(rd, d, ["Dropped.m"]);
    assert(d.a == 1.5 && d.s == "beam" && d.f && rd.done);

    // A field whose type changed is refused.
    static struct Changed { double a; int[2] b; string s; double[][] m; double f; }
    Changed c;
    auto rc = Bytes(o.data);
    readHeader(rc, "test", 1);
    assertThrown(read(rc, c));
}
