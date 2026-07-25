using System;

using NUnit.Framework.Constraints;

using Unity.Mathematics;

namespace BULA.mathProxies
{
    public struct fProxy2 {

        public float x;
        public float y;

        unsafe public fProxy this[int index] {
            get {
                if ((uint)index >= 2)
                    throw new System.ArgumentException("index must be between[0...1]");

                fixed (fProxy2* array = &this) { return ((float*)array)[index]; }
            }
            set {
                if ((uint)index >= 2)
                    throw new System.ArgumentException("index must be between[0...1]");

                fixed (float* array = &x) { array[index] = value; }
            }
        }

        
    }

    public struct fProxy3 {

        public float x;
        public float y;
        public float z;
        
        unsafe public fProxy this[int index] {
            get {
                if ((uint)index >= 3)
                    throw new System.ArgumentException("index must be between[0...2]");

                fixed (fProxy3* array = &this) { return ((float*)array)[index]; }
            }
            set {
                if ((uint)index >= 3)
                    throw new System.ArgumentException("index must be between[0...2]");

                fixed (float* array = &x) { array[index] = value; }
            }
        }
    }

    public struct fProxy4 {

        public float x;
        public float y;
        public float z;
        public float w;

        unsafe public fProxy this[int index] {
            get {
                if ((uint)index >= 4)
                    throw new System.ArgumentException("index must be between[0...3]");

                fixed (fProxy4* array = &this) { return ((float*)array)[index]; }
            }
            set {
                if ((uint)index >= 4)
                    throw new System.ArgumentException("index must be between[0...3]");

                fixed (float* array = &x) { array[index] = value; }
            }
        }

        // Component-wise + and * so templates can use fProxy4 as a width-4 SIMD accumulator
        // (fProxy4 acc; acc += load * load; ... acc.x+acc.y+acc.z+acc.w). These exist ONLY to compile
        // the template -- codegen substitutes fProxy4 -> float4/double4, whose built-in operators (and
        // real SIMD) are what actually run. See UnsafeOP.matVecDot for the pattern.
        public static fProxy4 operator +(fProxy4 a, fProxy4 b)
            => new fProxy4 { x = a.x + b.x, y = a.y + b.y, z = a.z + b.z, w = a.w + b.w };
        public static fProxy4 operator *(fProxy4 a, fProxy4 b)
            => new fProxy4 { x = a.x * b.x, y = a.y * b.y, z = a.z * b.z, w = a.w * b.w };
        public static fProxy4 operator -(fProxy4 a, fProxy4 b)
            => new fProxy4 { x = a.x - b.x, y = a.y - b.y, z = a.z - b.z, w = a.w - b.w };
        public static fProxy4 operator /(fProxy4 a, fProxy4 b)
            => new fProxy4 { x = a.x / b.x, y = a.y / b.y, z = a.z / b.z, w = a.w / b.w };

        // Scalar-broadcast * so templates can write `a * vec4Load` (the fused Krylov-kernel
        // pattern) -- real float4/double4 already support this natively; the stub just needs to
        // match it to compile the template. Never runs (see the note above).
        public static fProxy4 operator *(fProxy a, fProxy4 b)
            => new fProxy4 { x = a * b.x, y = a * b.y, z = a * b.z, w = a * b.w };
        public static fProxy4 operator *(fProxy4 a, fProxy b)
            => new fProxy4 { x = a.x * b, y = a.y * b, z = a.z * b, w = a.w * b };
    }

    public struct fProxy2x2 {

        public fProxy2 c0;
        public fProxy2 c1;

        unsafe public ref fProxy2 this[int index] {
            get {
                if ((uint)index >= 2)
                    throw new System.ArgumentException("index must be between[0...1]");

                fixed (fProxy2x2* array = &this) { return ref ((fProxy2*)array)[index]; }
            }
        }
    }

    public struct fProxy2x3
    {
        public fProxy2 c0;
        public fProxy2 c1;
        public fProxy2 c2;
                                         
        unsafe public ref fProxy2 this[int index] {
            get {
                if ((uint)index >= 3)
                    throw new System.ArgumentException("index must be between[0...2]");

                fixed (fProxy2x3* array = &this) { return ref ((fProxy2*)array)[index]; }
            }
        }
    }

    public struct fProxy2x4 {

        public fProxy2 c0;
        public fProxy2 c1;
        public fProxy2 c2;
        public fProxy2 c3;
                                         
        unsafe public ref fProxy2 this[int index] {
            get {
                if ((uint)index >= 4)
                    throw new System.ArgumentException("index must be between[0...3]");

                fixed (fProxy2x4* array = &this) { return ref ((fProxy2*)array)[index]; }
            }
        }
    }

    public struct fProxy3x3 {
        public fProxy3 c0;
        public fProxy3 c1;
        public fProxy3 c2;

        unsafe public ref fProxy3 this[int index] {
            get {
                if ((uint)index >= 3)
                    throw new System.ArgumentException("index must be between[0...2]");

                fixed (fProxy3x3* array = &this) { return ref ((fProxy3*)array)[index]; }
            }
        }
    }

    public struct fProxy3x2 {

        public fProxy3 c0;
        public fProxy3 c1;

        unsafe public ref fProxy3 this[int index] {
            get {
                if ((uint)index >= 2)
                    throw new System.ArgumentException("index must be between[0...1]");

                fixed (fProxy3x2* array = &this) { return ref ((fProxy3*)array)[index]; }
            }
        }
    }

    public struct fProxy3x4 {

        public fProxy3 c0;
        public fProxy3 c1;
        public fProxy3 c2;
        public fProxy3 c3;

        unsafe public ref fProxy3 this[int index] {
            get {
                if ((uint)index >= 4)
                    throw new System.ArgumentException("index must be between[0...3]");

                fixed (fProxy3x4* array = &this) { return ref ((fProxy3*)array)[index]; }
            }
        }
    }

    public struct fProxy4x4 {
        public fProxy4 c0;
        public fProxy4 c1;
        public fProxy4 c2;
        public fProxy4 c3;

        unsafe public ref fProxy4 this[int index] {
            get {
                if ((uint)index >= 4)
                    throw new System.ArgumentException("index must be between [0...3]");

                fixed (fProxy4x4* array = &this) { return ref ((fProxy4*)array)[index]; }
            }
        }
    }

    public struct fProxy4x2 {

        public fProxy4 c0;
        public fProxy4 c1;

        unsafe public ref fProxy4 this[int index] {
            get {
                if ((uint)index >= 2)
                    throw new System.ArgumentException("index must be between[0...1]");

                fixed (fProxy4x2* array = &this) { return ref ((fProxy4*)array)[index]; }
            }
        }
    }

    public struct fProxy4x3 {

        public fProxy4 c0;
        public fProxy4 c1;
        public fProxy4 c2;

        unsafe public ref fProxy4 this[int index] {
            get {
                if ((uint)index >= 3)
                    throw new System.ArgumentException("index must be between[0...2]");

                fixed (fProxy4x3* array = &this) { return ref ((fProxy4*)array)[index]; }
            }
        }
    }

}

