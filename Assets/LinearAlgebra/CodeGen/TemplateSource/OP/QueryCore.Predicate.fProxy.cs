#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

namespace LinearAlgebra
{
    // Internal generic bodies for Query's Group-A flat/scalar predicate ops (findFirst, count,
    // any, all, findAll). The public Query surface exposes these as concrete-shape overloads
    // (fProxyN / fProxyMxN) that forward here -- the array type T is fixed at the call site while
    // the predicate P stays generic. float/double emit distinct floatQueryCore/doubleQueryCore, so
    // the type-identical `<T,P>(in T, ref P)` signatures never collide (CS0111); Burst monomorphizes
    // both the forwarder and this body into the same machine code (zero perf downgrade).
    internal static partial class fProxyQueryCore
    {
        public static int findFirst<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafefProxyArray
            where P : struct, IfProxyPredicate
        {
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) return i;
            return -1;
        }

        public static int count<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafefProxyArray
            where P : struct, IfProxyPredicate
        {
            int c = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) c++;
            return c;
        }

        public static bool any<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafefProxyArray
            where P : struct, IfProxyPredicate
        {
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) return true;
            return false;
        }

        public static bool all<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafefProxyArray
            where P : struct, IfProxyPredicate
        {
            for (int i = 0; i < x.Data.Length; i++)
                if (!pred.Test(x.Data[i])) return false;
            return true;
        }

        public static int findAll<T, P>(in T x, ref P pred, ref Indices idx)
            where T : unmanaged, IUnsafefProxyArray
            where P : struct, IfProxyPredicate
        {
            if (idx.N < x.Data.Length)
                throw new System.ArgumentException("Query.findAll: idx.N must be >= x.Data.Length");
            int c = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) idx[c++] = i;
            return c;
        }
    }
}
