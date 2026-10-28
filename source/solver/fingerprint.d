/// Fingerprints: what a saved state was computed from, as name/value
/// pairs (the problem's inputs and settings, each in a text that changes
/// whenever it does), so a state is only resumed on the problem it
/// belongs to. Moved from pleaser's state.d.
module solver.fingerprint;

import std.format : format;

/// The first difference between a state's fingerprint and the current
/// one, as a sentence; null when they hold the same pairs (in any order).
string mismatch(const(string[2])[] saved, const(string[2])[] current) pure
{
    foreach (kv; current)
    {
        bool found = false;
        foreach (s; saved)
            if (s[0] == kv[0])
            {
                found = true;
                if (s[1] != kv[1])
                    return format("%s is %s now, %s in the state", kv[0], kv[1], s[1]);
            }
        if (!found)
            return format("%s (%s) is not in the state", kv[0], kv[1]);
    }
    foreach (s; saved)
    {
        bool found = false;
        foreach (kv; current)
            found |= kv[0] == s[0];
        if (!found)
            return format("the state has %s (%s), which this run doesn't", s[0], s[1]);
    }
    return null;
}

unittest
{
    const string[2][] st = [["lufs", "-16"], ["stem a", "x"]];
    assert(mismatch(st, st) is null);
    assert(mismatch(st, [st[1], st[0]]) is null);
    assert(mismatch(st, [["lufs", "-14"], ["stem a", "x"]]) == "lufs is -14 now, -16 in the state");
    assert(mismatch(st, st ~ ["stem b", "y"]) == "stem b (y) is not in the state");
    assert(mismatch(st, st[0 .. 1]) == "the state has stem a (x), which this run doesn't");
}
