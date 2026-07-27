using Unity.Collections.LowLevel.Unsafe;
//singularFile//
//alsoExpand[uint]// widens the IUnsafeiProxyArray copyReplace block below, so IUnsafeuintArray is
//generated the same way IUnsafeintArray/IUnsafeshortArray/IUnsafelongArray already are (this is a
//singular file, but singular files opt into alsoExpand exactly like a per-type file does - see
//TemplateConverter.Execute's singular-file loop).

/*
Useful for per-element operations which can happen on both vector or matrix types.
*/

namespace BULA
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

    public partial interface IAllocShortcuts
    {
        //+copyReplace
        public unsafe fProxyN fProxyVec(int n, bool uninit = false);

        public unsafe fProxyN fProxyTempVec(int n, bool uninit = false);

        public unsafe fProxyMxN fProxyMat(int m, int n, bool uninit = false);

        public unsafe fProxyMxN fProxyTempMat(int m, int n, bool uninit = false);
        //-copyReplace

        public unsafe boolN boolVec(int n, bool uninit = false);

        public unsafe boolN boolTempVec(int n, bool uninit = false);

        public unsafe boolMxN boolMat(int m, int n, bool uninit = false);

        public unsafe boolMxN boolTempMat(int m, int n, bool uninit = false);

    }

}