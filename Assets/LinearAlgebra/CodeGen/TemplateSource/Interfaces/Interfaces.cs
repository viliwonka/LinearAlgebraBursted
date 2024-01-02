using Unity.Collections.LowLevel.Unsafe;
//singularFile//

/*
Useful for per-element operations which can happen on both vector or matrix types.
*/

namespace LinearAlgebra
{
    //+copyReplace
    public interface IUnsafefProxyArray
    {
        public UnsafeList<fProxy> Data { get; }
    }
    //-copyReplace

    //+copyReplace
    public interface IUnsafeiProxyArray {
        public UnsafeList<iProxy> Data { get; }
    }
    //-copyReplace


    public interface IUnsafeBoolArray
    {
        public UnsafeList<bool> Data { get; }
    }

    public partial interface IArenaShortcuts
    {
        //+copyReplace
        public unsafe fProxyN fProxyVec(int n, bool uninit = false);

        public unsafe fProxyN tempfProxyVec(int n, bool uninit = false);

        public unsafe fProxyMxN fProxyMat(int m, int n, bool uninit = false);

        public unsafe fProxyMxN tempfProxyMat(int m, int n, bool uninit = false);
        //-copyReplace

        public unsafe boolN boolVec(int n, bool uninit = false);

        public unsafe boolN tempBoolVec(int n, bool uninit = false);

        public unsafe boolMxN boolMat(int m, int n, bool uninit = false);

        public unsafe boolMxN tempBoolMat(int m, int n, bool uninit = false);

    }

    public interface IMatrix<T> where T : unmanaged {

        ref T this[int row, int col] { get; }
        ref T this[int index] { get; }

        int M_Rows { get; }
        int N_Cols { get; }

        // Methods to copy data to and from the matrix
        void CopyTo(IMatrix<T> destination);
        void CopyFrom(IMatrix<T> source);

        // Read-only checks
        bool IsSquare { get; }

        // Other necessary properties and methods...
    }

}