#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

namespace LinearAlgebra
{
    // Internal generic bodies for Query's integer Group-A predicate ops. The public Query surface
    // exposes concrete-shape overloads (longN / longMxN) that forward here with the array type T
    // fixed and the predicate P generic. int/short/long emit distinct intQueryCore/shortQueryCore/
    // longQueryCore, so the type-identical `<T,P>(in T, ref P)` signatures never collide (CS0111);
    // Burst monomorphizes forwarder + body into the same machine code (zero perf downgrade).
    internal static partial class longQueryCore
    {
        public static int findFirst<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafelongArray
            where P : struct, IlongPredicate
        {
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) return i;
            return -1;
        }

        public static int count<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafelongArray
            where P : struct, IlongPredicate
        {
            int c = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) c++;
            return c;
        }

        public static bool any<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafelongArray
            where P : struct, IlongPredicate
        {
            for (int i = 0; i < x.Data.Length; i++)
                if (pred.Test(x.Data[i])) return true;
            return false;
        }

        public static bool all<T, P>(in T x, ref P pred)
            where T : unmanaged, IUnsafelongArray
            where P : struct, IlongPredicate
        {
            for (int i = 0; i < x.Data.Length; i++)
                if (!pred.Test(x.Data[i])) return false;
            return true;
        }

        public static int findAll<T, P>(in T x, ref P pred, ref Indices idx)
            where T : unmanaged, IUnsafelongArray
            where P : struct, IlongPredicate
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
