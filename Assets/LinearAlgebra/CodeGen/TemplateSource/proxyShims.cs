using Unity.Mathematics;

namespace LinearAlgebra
{
    public static class mathShims {

        public static fProxy NextFProxy(this ref Random random) {
            return (fProxy)random.NextFloat();
        }

        public static fProxy NextFProxy(this ref Random random, fProxy min, fProxy max) {
            return (fProxy)random.NextFloat(min, max);
        }

        public static iProxy NextIProxy(this ref Random random) {
            return (iProxy)random.NextInt();
        }

        public static iProxy NextIProxy(this ref Random random, iProxy min, iProxy max) {
            return (iProxy)random.NextInt(min, max);
        }
    }
}