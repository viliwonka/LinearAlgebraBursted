using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Concrete (NOT codegen'd) regression pins + cross-type contrasts for the Hash class. Deliberately
// hand-authored and run on the MANAGED main thread (no Burst job): the float bit-pattern pins
// (-0.0 vs +0.0, distinct NaN payloads) must not be exposed to Burst's FloatMode folding, which
// could canonicalize -0.0/NaN and defeat the very thing these tests verify (Hash reads raw bytes,
// not IEEE-equal values). Special float bit patterns are built via math.asfloat(rawBits) so the
// stored bytes are exactly the ones the reference oracle hashed.
//
// The expected uint constants were cross-checked against two independent xxHash32 implementations
// (a standalone .NET console port and a JS/BigInt port from the algorithm description) that agree
// bit-for-bit -- so a mismatch here is a real regression in Hash, not a fragile golden value.
public class HashSourceTests
{
    // hash of a zero-length buffer (byteLength == 0): defined, deterministic, and seed-DEPENDENT
    // (not a degenerate constant). Element type is irrelevant when the byte length is 0.
    [Test]
    public void EmptyHashPinned()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var e = arena.intVec(0);
            Assert.AreEqual(46947589u, Hash.hash(in e, 0u));
            Assert.AreEqual(2839904920u, Hash.hash(in e, 12345u));
            Assert.AreNotEqual(Hash.hash(in e, 0u), Hash.hash(in e, 12345u)); // seed-dependent
        }
        finally { arena.Dispose(); }
    }

    // Cross-type independence: {1,2,3} as int32 (4 bytes/elem) and as short16 (2 bytes/elem) are
    // DIFFERENT byte streams, so they hash differently. Hashes are per-byte-representation, NOT
    // per-logical-value -- this pins that fact precisely rather than asserting a vague inequality.
    [Test]
    public void CrossTypeIntVsShortPinned()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var vi = arena.intVec(3);
            vi[0] = 1; vi[1] = 2; vi[2] = 3;
            uint hi = Hash.hash(in vi, 0u);
            Assert.AreEqual(525831304u, hi);

            var vs = arena.shortVec(3);
            vs[0] = (short)1; vs[1] = (short)2; vs[2] = (short)3;
            uint hs = Hash.hash(in vs, 0u);
            Assert.AreEqual(113706251u, hs);

            Assert.AreNotEqual(hi, hs); // same logical values, different byte width -> different hash
        }
        finally { arena.Dispose(); }
    }

    // Length sensitivity: {1,2} vs {1,2,0} (int32, same seed) hash differently.
    [Test]
    public void LengthSensitivityPinned()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v2 = arena.intVec(2);
            v2[0] = 1; v2[1] = 2;
            uint h2 = Hash.hash(in v2, 0u);
            Assert.AreEqual(1762362331u, h2);

            var v3 = arena.intVec(3);
            v3[0] = 1; v3[1] = 2; v3[2] = 0;
            uint h3 = Hash.hash(in v3, 0u);
            Assert.AreEqual(4062492784u, h3);

            Assert.AreNotEqual(h2, h3);
        }
        finally { arena.Dispose(); }
    }

    // Float bit-hashing caveat #1: -0.0f and +0.0f are numerically EQUAL (-0.0f == 0.0f) but their
    // bit patterns differ in the sign bit -> different hashes.
    [Test]
    public void FloatSignedZeroPinned()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var neg = arena.floatVec(1);
            neg[0] = math.asfloat(0x80000000u); // -0.0f, exact bits
            uint hNeg = Hash.hash(in neg, 0u);
            Assert.AreEqual(1509677505u, hNeg);

            var pos = arena.floatVec(1);
            pos[0] = math.asfloat(0x00000000u); // +0.0f, exact bits
            uint hPos = Hash.hash(in pos, 0u);
            Assert.AreEqual(148298089u, hPos);

            Assert.IsTrue(neg[0] == pos[0]);       // numerically equal (-0.0f == +0.0f)
            Assert.AreNotEqual(hNeg, hPos);        // but hash differently (raw bytes differ)
        }
        finally { arena.Dispose(); }
    }

    // Float bit-hashing caveat #2: two NaNs with different raw payloads hash differently. Built from
    // explicit bits (NOT float.NaN twice, which would be the SAME payload and pass trivially).
    [Test]
    public void FloatNaNPayloadPinned()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var nanA = arena.floatVec(1);
            nanA[0] = math.asfloat(0x7fc00000u);
            uint hA = Hash.hash(in nanA, 0u);
            Assert.AreEqual(2181731943u, hA);

            var nanB = arena.floatVec(1);
            nanB[0] = math.asfloat(0x7fc00001u); // different payload low bit
            uint hB = Hash.hash(in nanB, 0u);
            Assert.AreEqual(2143227415u, hB);

            Assert.IsTrue(float.IsNaN(nanA[0]) && float.IsNaN(nanB[0])); // both are NaN
            Assert.AreNotEqual(hA, hB);                                  // but hash differently
        }
        finally { arena.Dispose(); }
    }

    // Sign-bit avalanche on an ordinary float: 1.5f vs -1.5f hash differently.
    [Test]
    public void FloatNegationPinned()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var pos = arena.floatVec(1);
            pos[0] = 1.5f;
            Assert.AreEqual(2849136462u, Hash.hash(in pos, 0u));

            var neg = arena.floatVec(1);
            neg[0] = -1.5f;
            Assert.AreEqual(714977028u, Hash.hash(in neg, 0u));

            Assert.AreNotEqual(Hash.hash(in pos, 0u), Hash.hash(in neg, 0u));
        }
        finally { arena.Dispose(); }
    }

    // combine is order-sensitive: combine(a,b) != combine(b,a) for ordinary small distinct values.
    // (Equality would need (a-b) == 0 mod 2^31; 1/2 and 5/100 are safely away from that boundary.)
    [Test]
    public void CombineNonCommutativePinned()
    {
        Assert.AreEqual(468815706u, Hash.combine(1u, 2u));
        Assert.AreEqual(1818560429u, Hash.combine(2u, 1u));
        Assert.AreNotEqual(Hash.combine(1u, 2u), Hash.combine(2u, 1u));

        Assert.AreEqual(2523756307u, Hash.combine(5u, 100u));
        Assert.AreEqual(2500008387u, Hash.combine(100u, 5u));
        Assert.AreNotEqual(Hash.combine(5u, 100u), Hash.combine(100u, 5u));
    }
}
