//alsoExpand[uint]// widens this file's default int/short/long rotation to include uint, so the
//int-family determinism group covers all four integer element types the library generates for
//(Blas.dot has a uint instantiation; Norms/Stats and Rand.nextUniformInPlace deliberately do not
//-- see the skipFor[u]/emitFor[u] guards below).

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of the determinism conformance harness's int-family group (row 25):
    // integer-state xorshift + `* /` only, so this whole group is bit-exact by construction (no
    // sqrt, no DetMath). See DeterminismDirect.fProxy.cs's header for the shared job/case-method
    // convention and docs/dev/spec-determinism-conformance-harness.md for the frozen op/group/root
    // hash contract.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetIntFamilyCoreJobIProxy : IJob
    {
        public iProxyN a, b;
        public Pivot pivot;
        public iProxyN randDest;

        public NativeArray<uint> HashOut; // 3 slots

        public unsafe void Execute()
        {
            iProxy dotResult = Blas.dot(a, b);
            HashOut[0] = Hash.hash((byte*)&dotResult, sizeof(iProxy));

            pivot.Swap(0, 3);
            pivot.Swap(1, 5);
            pivot.Swap(2, 6);
            HashOut[1] = DetHash.CombinePivot(0u, in pivot);

            HashOut[2] = Hash.hash(in randDest);
        }
    }

    //+skipFor[u]
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetIntFamilyNormsStatsJobIProxy : IJob
    {
        public iProxyN vec;

        public NativeArray<uint> HashOut; // 1 slot

        public void Execute()
        {
            long l1 = Norms.L1(in vec);
            long lInf = Norms.LInf(in vec);
            long sum = Stats.sum(in vec);
            uint h = DetHash.Combine(0u, l1);
            h = DetHash.Combine(h, lInf);
            h = DetHash.Combine(h, sum);
            HashOut[0] = h;
        }
    }
    //-skipFor

    public static partial class DeterminismInt
    {
        public static (string id, uint hash)[] Case_IntFamilyCoreIProxy()
        {
            var rng = new Random(2654435761u ^ 0x0019u);

            const int n = 48;
            // Non-negative range: this case runs for uint too (unlike the norms/stats job below,
            // which is skipped for uint), and a negative iProxy literal wraps to a huge value under
            // unsigned arithmetic, which would violate Rand.nextUniformInPlace's min <= max contract.
            var a = new iProxyN(n, Allocator.Persistent);
            var b = new iProxyN(n, Allocator.Persistent);
            // Small range: n=48 values in [0,10) keeps the worst-case dot-product sum (4800) well
            // inside `short`'s range (Blas.dot accumulates in the element's own width, unwidened).
            // Rand.nextUniformInPlace has no uint instantiation (RandomOP.iProxy.cs does not opt
            // into the uint alsoExpand widening) -- fall back to Unity.Mathematics.Random.NextUInt
            // directly for that one slot.
            //+skipFor[u]
            Rand.nextUniformInPlace(ref rng, ref a, (iProxy)0, (iProxy)10);
            Rand.nextUniformInPlace(ref rng, ref b, (iProxy)0, (iProxy)10);
            //-skipFor
            //+emitFor[u]
            //!for (int i = 0; i < n; i++) { a[i] = rng.NextUInt(0u, 10u); b[i] = rng.NextUInt(0u, 10u); }
            //-emitFor

            var pivot = new Pivot(8, Allocator.Persistent);

            var randDest = new iProxyN(n, Allocator.Persistent);
            //+skipFor[u]
            Rand.nextUniformInPlace(ref rng, ref randDest, (iProxy)0, (iProxy)1000);
            //-skipFor
            //+emitFor[u]
            //!for (int i = 0; i < n; i++) randDest[i] = rng.NextUInt(0u, 1000u);
            //-emitFor

            var hashOut = new NativeArray<uint>(3, Allocator.Persistent);
            var job = new DetIntFamilyCoreJobIProxy { a = a, b = b, pivot = pivot, randDest = randDest, HashOut = hashOut };
            job.Run();

            var result = new[]
            {
                ("int-family/dot.iProxy.n48", hashOut[0]),
                ("int-family/pivot-roundtrip.iProxy", hashOut[1]),
                ("int-family/randomFill.iProxy.n48", hashOut[2]),
            };
            hashOut.Dispose();
            pivot.Dispose();
            a.Dispose(); b.Dispose(); randDest.Dispose();
            return result;
        }

        //+skipFor[u]
        public static (string id, uint hash)[] Case_IntFamilyNormsStatsIProxy()
        {
            var rng = new Random(2654435761u ^ 0x001Au);

            const int n = 48;
            var vec = new iProxyN(n, Allocator.Persistent);
            Rand.nextUniformInPlace(ref rng, ref vec, (iProxy)(-50), (iProxy)50);

            var hashOut = new NativeArray<uint>(1, Allocator.Persistent);
            var job = new DetIntFamilyNormsStatsJobIProxy { vec = vec, HashOut = hashOut };
            job.Run();

            var result = new[] { ("int-family/norms-stats.iProxy.n48", hashOut[0]) };
            hashOut.Dispose();
            vec.Dispose();
            return result;
        }
        //-skipFor
    }
}
