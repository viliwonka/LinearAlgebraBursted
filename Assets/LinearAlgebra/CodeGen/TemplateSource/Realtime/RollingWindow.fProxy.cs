#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Stats;

namespace LinearAlgebra.Realtime
{
    /// <summary>
    /// Fixed-capacity sliding window of feature vectors — the realtime front-end that makes the
    /// library's matrix ops usable in a per-frame loop. Internally a ring buffer of
    /// <c>Capacity</c> rows × <c>Features</c> columns; <see cref="Push"/> overwrites the oldest
    /// sample once full (O(Features), no allocation, no shifting).
    ///
    /// Indexed oldest→newest (<c>this[0]</c> = oldest retained sample, <c>this[Count-1]</c> = newest).
    /// <see cref="AsMatrix(ref fProxyMxN)"/> materializes the window time-ordered into a contiguous
    /// Count×Features matrix so it can feed any existing kernel (covariance → eigendecomposition = PCA;
    /// AsMatrix + qrDirectSolve = least-squares trajectory fit; <see cref="Mean"/> = moving average).
    ///
    /// Create with <c>arena.fProxyRollingWindow(capacity, features)</c>; the backing buffer is a
    /// persistent arena allocation that lives until the arena is disposed. fProxy-only.
    /// </summary>
    public struct fProxyRollingWindow
    {
        private fProxyMxN _buffer;   // Capacity rows × Features cols, ring storage
        private int _capacity;
        private int _features;
        private int _head;           // row the NEXT Push writes (0..Capacity-1)
        private int _count;          // number of valid samples (0..Capacity)

        public int Capacity => _capacity;
        public int Features => _features;
        public int Count => _count;
        public bool IsFull => _count == _capacity;
        public bool IsEmpty => _count == 0;

        /// <summary>Internal — use <c>arena.fProxyRollingWindow(capacity, features)</c>.</summary>
        internal fProxyRollingWindow(in fProxyMxN buffer, int capacity, int features)
        {
            _buffer = buffer;
            _capacity = capacity;
            _features = features;
            _head = 0;
            _count = 0;
        }

        // Row of the ring holding the oldest retained sample. Before the buffer fills, samples sit in
        // rows 0.._count-1 (oldest = 0); once full the write head sits on the oldest sample.
        private int OldestRow => _count < _capacity ? 0 : _head;

        // Map logical index i (0 = oldest .. Count-1 = newest) to the physical ring row.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int RingRow(int i) => (OldestRow + i) % _capacity;

        /// <summary>
        /// Append a sample (length must equal Features). When the window is full this overwrites the
        /// oldest sample; the logical order is preserved. O(Features), no allocation.
        /// </summary>
        public void Push(in fProxyN sample)
        {
            if (sample.N != _features)
                throw new ArgumentException("RollingWindow.Push: sample length must equal Features");

            int row = _head;
            for (int c = 0; c < _features; c++)
                _buffer[row, c] = sample[c];

            _head++;
            if (_head == _capacity) _head = 0;
            if (_count < _capacity) _count++;
        }

        /// <summary>Value of feature f in the i-th retained sample (i: 0 = oldest .. Count-1 = newest).</summary>
        public fProxy this[int i, int f]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Assume.IndexInsideBounds(new int2(_count, _features), new int2(i, f));
#endif
                return _buffer[RingRow(i), f];
            }
        }

        /// <summary>Copies the i-th retained sample (0 = oldest) into dest (length Features).</summary>
        public void GetSample(int i, ref fProxyN dest)
        {
            if (dest.N != _features)
                throw new ArgumentException("RollingWindow.GetSample: dest length must equal Features");

            int row = RingRow(i);
            for (int c = 0; c < _features; c++)
                dest[c] = _buffer[row, c];
        }

        /// <summary>Logically empties the window (Count → 0). The backing buffer is not freed or zeroed.</summary>
        public void Clear()
        {
            _head = 0;
            _count = 0;
        }

        /// <summary>
        /// Writes the retained samples, time-ordered (row 0 = oldest), into dest. dest must be
        /// exactly Count × Features. Zero-alloc — reuse one dest matrix across frames.
        /// </summary>
        public void AsMatrix(ref fProxyMxN dest)
        {
            if (dest.M_Rows != _count || dest.N_Cols != _features)
                throw new ArgumentException("RollingWindow.AsMatrix: dest must be Count x Features");

            for (int i = 0; i < _count; i++)
            {
                int row = RingRow(i);
                for (int c = 0; c < _features; c++)
                    dest[i, c] = _buffer[row, c];
            }
        }

        /// <summary>
        /// Allocating AsMatrix: returns a fresh Count×Features time-ordered matrix from the arena's
        /// TEMP pool (reclaimed by ClearTemp), so per-frame use leaks nothing.
        /// </summary>
        public fProxyMxN AsMatrix()
        {
            var m = _buffer.tempfProxyMat(_count, _features);
            AsMatrix(ref m);
            return m;
        }

        /// <summary>
        /// Per-feature mean over the retained samples (the moving average). dest length must equal
        /// Features. Zero-alloc. Throws if the window is empty.
        /// </summary>
        public void Mean(ref fProxyN dest)
        {
            if (_count == 0)
                throw new InvalidOperationException("RollingWindow.Mean: window is empty");
            if (dest.N != _features)
                throw new ArgumentException("RollingWindow.Mean: dest length must equal Features");

            for (int c = 0; c < _features; c++)
                dest[c] = (fProxy)0;

            for (int i = 0; i < _count; i++)
            {
                int row = RingRow(i);
                for (int c = 0; c < _features; c++)
                    dest[c] += _buffer[row, c];
            }
            for (int c = 0; c < _features; c++)
                dest[c] /= (fProxy)_count;
        }

        /// <summary>Allocating moving average — a fresh Features vector from the arena TEMP pool.</summary>
        public fProxyN Mean()
        {
            var v = _buffer.tempfProxyVec(_features);
            Mean(ref v);
            return v;
        }

        /// <summary>
        /// Sample covariance of the features over the window (Features × Features, ÷(Count-1)), written
        /// into dest. Requires Count ≥ 2. Zero-alloc apart from one internal temp matrix (TEMP pool).
        /// Pair with <c>Eigen.eigenDecomposition</c> on the result for realtime PCA / dominant motion.
        /// </summary>
        public void Covariance(ref fProxyMxN dest)
        {
            if (_count < 2)
                throw new InvalidOperationException("RollingWindow.Covariance: requires at least 2 samples");
            if (dest.M_Rows != _features || dest.N_Cols != _features)
                throw new ArgumentException("RollingWindow.Covariance: dest must be Features x Features");

            // Time-order into a temp matrix, then reuse the StatsOP covariance core.
            var m = _buffer.tempfProxyMat(_count, _features);
            AsMatrix(ref m);
            fProxyStats_OP.covarianceInto(in m, ref dest);
        }

        /// <summary>Allocating covariance — a fresh Features×Features matrix from the arena TEMP pool.</summary>
        public fProxyMxN Covariance()
        {
            var c = _buffer.tempfProxyMat(_features, _features);
            Covariance(ref c);
            return c;
        }
    }

    /// <summary>Arena factory for <see cref="fProxyRollingWindow"/>.</summary>
    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a rolling window holding up to <paramref name="capacity"/> samples of
        /// <paramref name="features"/> features each. The backing buffer is a persistent arena
        /// allocation (lives until the arena is disposed); the window starts empty.
        /// </summary>
        public static fProxyRollingWindow fProxyRollingWindow(this ref Arena arena, int capacity, int features)
        {
            if (capacity < 1)
                throw new ArgumentException("fProxyRollingWindow: capacity must be >= 1");
            if (features < 1)
                throw new ArgumentException("fProxyRollingWindow: features must be >= 1");

            var buffer = arena.fProxyMat(capacity, features);
            return new fProxyRollingWindow(in buffer, capacity, features);
        }
    }
}
