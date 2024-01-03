using Unity.Collections.LowLevel.Unsafe;
//singularFile//

/*
Useful for per-element operations which can happen on both vector or matrix types.
*/

namespace LinearAlgebra
{
    
    public interface IUnsafefloatArray
    {
        public UnsafeList<float> Data { get; }
    }
    
    public interface IUnsafedoubleArray
    {
        public UnsafeList<double> Data { get; }
    }
    

    
    public interface IUnsafeintArray {
        public UnsafeList<int> Data { get; }
    }
    
    public interface IUnsafeshortArray {
        public UnsafeList<short> Data { get; }
    }
    
    public interface IUnsafelongArray {
        public UnsafeList<long> Data { get; }
    }
    


    public interface IUnsafeBoolArray
    {
        public UnsafeList<bool> Data { get; }
    }

    public partial interface IArenaShortcuts
    {
        
        public unsafe floatN floatVec(int n, bool uninit = false);

        public unsafe floatN tempfloatVec(int n, bool uninit = false);

        public unsafe floatMxN floatMat(int m, int n, bool uninit = false);

        public unsafe floatMxN tempfloatMat(int m, int n, bool uninit = false);
        
        public unsafe doubleN doubleVec(int n, bool uninit = false);

        public unsafe doubleN tempdoubleVec(int n, bool uninit = false);

        public unsafe doubleMxN doubleMat(int m, int n, bool uninit = false);

        public unsafe doubleMxN tempdoubleMat(int m, int n, bool uninit = false);
        

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