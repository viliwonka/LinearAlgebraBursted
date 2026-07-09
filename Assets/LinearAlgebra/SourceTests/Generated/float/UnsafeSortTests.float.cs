using System;

using LinearAlgebra;
using LinearAlgebra.Internal;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

// Direct kernel tests for UnsafeOP.sortByKeyAscending (in-place ASCENDING heapsort of parallel
// key/value arrays -- LinearAlgebra.Internal.UnsafeOP, UnsafeOP.float.cs). Added per
// docs/spec-shipped-feature.md pillar 3 ("New Blas/UnsafeOP kernels get DIRECT tests against a plain
// scalar reference implementation, not just indirect coverage through callers"). This kernel powers
// LP.ladBR's large-candidate fast path (LP.BarrodaleRoberts.float.cs, gated by
// BR_CAND_SORT_THRESHOLD = 256), so before this file the kernel had ZERO direct coverage.
//
// TEMPLATE (not hand-written) test: UnsafeOP and sortByKeyAscending are both `public` (the class only
// lives in the LinearAlgebra.Internal namespace), and template tests already reach it directly with
// raw pointers inside Burst jobs -- see QRCPDowndateTests.float.cs / LUTests.float.cs calling
// LinearAlgebra.Internal.UnsafeOP.axpy/maxAbs. There is no InternalsVisibleTo barrier here (that only
// forces the hand-written SourceTests route for genuinely `internal` members such as
// ladFrischNewtonCore), so a template test is used to get automatic float+double coverage.
//
// The sort is NOT stable (heapsort never is): candidates with EXACTLY EQUAL keys may reorder relative
// to a stable sort. So the reference checks below assert (a) keys come out ascending and equal to a
// plain scalar reference sort of the SAME multiset (exact equality -- sorting only permutes values, so
// no epsilon and identical for both dtypes), and (b) the parallel int payload rode along correctly:
// `val` is a permutation of the input indices and key[i] == originalKey[val[i]] for every i. Tie ORDER
// is deliberately NOT asserted.
public class floatUnsafeSortTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public unsafe struct TestJob : IJob
    {
        public enum TestType
        {
            RandomVariousN,   // random keys at n = 1, 2, 7, 64, 1000
            DuplicateKeys,    // small-range keys (many exact ties) -- keys ascend, multiset preserved
            EmptyAndSingle,   // n = 0 (no-op, buffers untouched) and n = 1
            AlreadySorted,    // ascending input stays ascending, payload identity preserved
            ReverseSorted,    // strictly descending input -> fully reversed
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] scenario/n, [2] index/expected, [3] extra
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.RandomVariousN: RandomVariousN(); break;
                case TestType.DuplicateKeys: DuplicateKeys(); break;
                case TestType.EmptyAndSingle: EmptyAndSingle(); break;
                case TestType.AlreadySorted: AlreadySorted(); break;
                case TestType.ReverseSorted: ReverseSorted(); break;
            }
        }

        // Random keys at a spread of sizes -- 1, 2, 7 exercise the tiny/odd shapes; 64 and especially
        // 1000 exercise the sizes at which LP.ladBR actually takes this path (nCand > 256).
        void RandomVariousN()
        {
            int* sizes = stackalloc int[5] { 1, 2, 7, 64, 1000 };
            for (int s = 0; s < 5; s++)
            {
                int n = sizes[s];
                var rng = new Unity.Mathematics.Random((uint)(n * 2654435761u + 12345u));
                var key = new floatN(n, Allocator.Temp);
                var val = new NativeArray<int>(n, Allocator.Temp);
                for (int i = 0; i < n; i++) { key[i] = rng.NextFloat(-(float)1000, (float)1000); val[i] = i; }
                RunAndVerify(key, val, n, 100 + s);
                key.Dispose(); val.Dispose();
            }
        }

        // Small key range (0..9) over n = 300 -> guaranteed many exact ties. Verifies keys ascend and
        // the key->value multiset is preserved WITHOUT asserting tie order (the sort is unstable).
        void DuplicateKeys()
        {
            int n = 300;
            var rng = new Unity.Mathematics.Random(777u);
            var key = new floatN(n, Allocator.Temp);
            var val = new NativeArray<int>(n, Allocator.Temp);
            for (int i = 0; i < n; i++) { key[i] = (float)rng.NextInt(0, 10); val[i] = i; }
            RunAndVerify(key, val, n, 200);
            key.Dispose(); val.Dispose();
        }

        // n = 0 must be a clean no-op (no writes, no crash); n = 1 is the trivial single-element case.
        void EmptyAndSingle()
        {
            // n = 0: allocate a length-1 buffer with a sentinel, sort n=0, assert it is UNTOUCHED.
            {
                var key = new floatN(1, Allocator.Temp);
                var val = new NativeArray<int>(1, Allocator.Temp);
                key[0] = (float)42; val[0] = 7;
                unsafe { UnsafeOP.sortByKeyAscending(key.Data.Ptr, (int*)NativeArrayUnsafeUtility.GetUnsafePtr(val), 0); }
                if (key[0] != (float)42) Rec(300, 0, 0);
                if (val[0] != 7) Rec(301, 0, 0);
                key.Dispose(); val.Dispose();
            }
            // n = 1: single element, already sorted, payload preserved.
            {
                var key = new floatN(1, Allocator.Temp);
                var val = new NativeArray<int>(1, Allocator.Temp);
                key[0] = (float)(-5); val[0] = 0;
                RunAndVerify(key, val, 1, 302);
                key.Dispose(); val.Dispose();
            }
        }

        // Ascending input must stay ascending and (since keys are strictly increasing here) the payload
        // must be the identity permutation.
        void AlreadySorted()
        {
            int n = 128;
            var key = new floatN(n, Allocator.Temp);
            var val = new NativeArray<int>(n, Allocator.Temp);
            for (int i = 0; i < n; i++) { key[i] = (float)i; val[i] = i; }
            RunAndVerify(key, val, n, 400);
            // strictly increasing distinct keys -> stable-or-not, the permutation is forced to identity.
            for (int i = 0; i < n; i++) if (val[i] != i) { Rec(401, i, val[i]); break; }
            key.Dispose(); val.Dispose();
        }

        // Strictly descending input -> fully reversed; payload val[i] must equal (n-1-i).
        void ReverseSorted()
        {
            int n = 128;
            var key = new floatN(n, Allocator.Temp);
            var val = new NativeArray<int>(n, Allocator.Temp);
            for (int i = 0; i < n; i++) { key[i] = (float)(n - i); val[i] = i; }
            RunAndVerify(key, val, n, 500);
            for (int i = 0; i < n; i++) if (val[i] != n - 1 - i) { Rec(501, i, val[i]); break; }
            key.Dispose(); val.Dispose();
        }

        // Snapshots the input, runs the kernel, and checks every property against a plain scalar
        // reference sort of the same multiset. `code` tags the scenario for the failure diagnostics.
        void RunAndVerify(floatN key, NativeArray<int> val, int n, int code)
        {
            // snapshot the original key[] (indexed by the ORIGINAL position == the val payload value)
            var orig = new floatN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) orig[i] = key[i];

            // plain scalar reference: selection-sort a copy of the keys ascending.
            var refKey = new floatN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) refKey[i] = key[i];
            for (int i = 0; i < n - 1; i++)
            {
                int mIdx = i;
                for (int j = i + 1; j < n; j++) if (refKey[j] < refKey[mIdx]) mIdx = j;
                if (mIdx != i) { float t = refKey[i]; refKey[i] = refKey[mIdx]; refKey[mIdx] = t; }
            }

            unsafe { UnsafeOP.sortByKeyAscending(key.Data.Ptr, (int*)NativeArrayUnsafeUtility.GetUnsafePtr(val), n); }

            // (1) keys non-decreasing
            for (int i = 1; i < n; i++)
                if (key[i - 1] > key[i]) { Rec(code, i, 0); goto done; }

            // (2) keys equal the scalar reference sort ELEMENTWISE (exact -- sorting only permutes;
            //     the sorted sequence of a multiset is unique regardless of stability)
            for (int i = 0; i < n; i++)
                if (key[i] != refKey[i]) { Rec(code + 10000, i, 0); goto done; }

            // (3) val is a permutation of [0,n): every index appears exactly once
            var seen = new NativeArray<int>(n > 0 ? n : 1, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                int v = val[i];
                if (v < 0 || v >= n) { Rec(code + 20000, i, v); seen.Dispose(); goto done; }
                if (seen[v] != 0) { Rec(code + 30000, i, v); seen.Dispose(); goto done; }
                seen[v] = 1;
            }
            seen.Dispose();

            // (4) the parallel payload rode along: key[i] == orig[val[i]] (the pair moved together --
            //     this is the ONLY value-order check, and it tolerates ties)
            for (int i = 0; i < n; i++)
                if (key[i] != orig[val[i]]) { Rec(code + 40000, i, val[i]); goto done; }

        done:
            orig.Dispose(); refKey.Dispose();
        }

        // Record-only (Burst-safe): the first failure is latched into Fail[] and the managed
        // TestCaseSource wrapper below turns a nonzero Fail[0] into an Assert.Fail. No throw inside the
        // job (so every Allocator.Temp buffer still disposes on its normal path).
        void Rec(int code, int a, int b)
        {
            if (Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = (float)code; Fail[2] = (float)a; Fail[3] = (float)b; }
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void UnsafeSortTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: failCode {fail[1]}, index/val {fail[2]}, extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: failCode {fail[1]}, index/val {fail[2]}, extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }
}
