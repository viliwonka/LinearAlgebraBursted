using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;
//singularFile//
namespace LinearAlgebra
{
    // Allocation helper
    [StructLayout(LayoutKind.Sequential)]
    public partial struct Arena : System.IDisposable {

        public int AllocationsCount => 
            
            floatVectors.Length + floatMatrices.Length
            +
            doubleVectors.Length + doubleMatrices.Length
            
            +
            
            intVectors.Length + intMatrices.Length
            +
            shortVectors.Length + shortMatrices.Length
            +
            longVectors.Length + longMatrices.Length
            
        ;

        public int TempAllocationsCount => 
            
            tempfloatVectors.Length + tempfloatMatrices.Length
            +
            tempdoubleVectors.Length + tempdoubleMatrices.Length
            
            +
            
            tempintVectors.Length + tempintMatrices.Length
            +
            tempshortVectors.Length + tempshortMatrices.Length
            +
            templongVectors.Length + templongMatrices.Length
            
        ;

        public int AllAllocationsCount => AllocationsCount + TempAllocationsCount;

        public Allocator Allocator;
        public bool Initialized { get; private set; }

        private UnsafeList<boolN> BoolVectors;
        private UnsafeList<boolMxN> BoolMatrices;
        private UnsafeList<boolN> TempBoolVectors;
        private UnsafeList<boolMxN> TempBoolMatrices;
        
        public Arena(Allocator allocator) {

            Initialized = true;
            Allocator = allocator;
            
            
            floatVectors = new UnsafeList<floatN>(8, Allocator);
            floatMatrices = new UnsafeList<floatMxN>(8, Allocator);
            tempfloatVectors = new UnsafeList<floatN>(8, Allocator);
            tempfloatMatrices = new UnsafeList<floatMxN>(8, Allocator);
            
            doubleVectors = new UnsafeList<doubleN>(8, Allocator);
            doubleMatrices = new UnsafeList<doubleMxN>(8, Allocator);
            tempdoubleVectors = new UnsafeList<doubleN>(8, Allocator);
            tempdoubleMatrices = new UnsafeList<doubleMxN>(8, Allocator);
            

            
            intVectors = new UnsafeList<intN>(8, Allocator);
            intMatrices = new UnsafeList<intMxN>(8, Allocator);
            tempintVectors = new UnsafeList<intN>(8, Allocator);
            tempintMatrices = new UnsafeList<intMxN>(8, Allocator);
            
            shortVectors = new UnsafeList<shortN>(8, Allocator);
            shortMatrices = new UnsafeList<shortMxN>(8, Allocator);
            tempshortVectors = new UnsafeList<shortN>(8, Allocator);
            tempshortMatrices = new UnsafeList<shortMxN>(8, Allocator);
            
            longVectors = new UnsafeList<longN>(8, Allocator);
            longMatrices = new UnsafeList<longMxN>(8, Allocator);
            templongVectors = new UnsafeList<longN>(8, Allocator);
            templongMatrices = new UnsafeList<longMxN>(8, Allocator);
            

            BoolVectors = new UnsafeList<boolN>(2, Allocator);
            BoolMatrices = new UnsafeList<boolMxN>(2, Allocator);

            TempBoolVectors = new UnsafeList<boolN>(2, Allocator);
            TempBoolMatrices = new UnsafeList<boolMxN>(2, Allocator);
        }

        public void Clear() {

            
            for (int i = 0; i < floatVectors.Length; i++)
                floatVectors[i].Dispose();
            floatVectors.Clear();

            for(int i = 0; i < floatMatrices.Length; i++)
                floatMatrices[i].Dispose();
            floatMatrices.Clear();
            
            for (int i = 0; i < doubleVectors.Length; i++)
                doubleVectors[i].Dispose();
            doubleVectors.Clear();

            for(int i = 0; i < doubleMatrices.Length; i++)
                doubleMatrices[i].Dispose();
            doubleMatrices.Clear();
            

            
            for (int i = 0; i < intVectors.Length; i++)
                intVectors[i].Dispose();
            intVectors.Clear();

            for (int i = 0; i < intMatrices.Length; i++)
                intMatrices[i].Dispose();
            intMatrices.Clear();
            
            for (int i = 0; i < shortVectors.Length; i++)
                shortVectors[i].Dispose();
            shortVectors.Clear();

            for (int i = 0; i < shortMatrices.Length; i++)
                shortMatrices[i].Dispose();
            shortMatrices.Clear();
            
            for (int i = 0; i < longVectors.Length; i++)
                longVectors[i].Dispose();
            longVectors.Clear();

            for (int i = 0; i < longMatrices.Length; i++)
                longMatrices[i].Dispose();
            longMatrices.Clear();
            

            for (int i = 0; i < BoolVectors.Length; i++)
                BoolVectors[i].Dispose();
            BoolVectors.Clear();

            for (int i = 0; i < BoolMatrices.Length; i++)
                BoolMatrices[i].Dispose();
            BoolMatrices.Clear();

            ClearTemp();
        }

        /// <summary>
        /// dispose only temporary allocations, produced from operations
        /// </summary>
        public void ClearTemp()
        {
            
            for (int i = 0; i < tempfloatVectors.Length; i++)
                tempfloatVectors[i].Dispose();
            tempfloatVectors.Clear();

            for (int i = 0; i < tempfloatMatrices.Length; i++)
                tempfloatMatrices[i].Dispose();
            tempfloatMatrices.Clear();
            
            for (int i = 0; i < tempdoubleVectors.Length; i++)
                tempdoubleVectors[i].Dispose();
            tempdoubleVectors.Clear();

            for (int i = 0; i < tempdoubleMatrices.Length; i++)
                tempdoubleMatrices[i].Dispose();
            tempdoubleMatrices.Clear();
            

            
            for (int i = 0; i < tempintVectors.Length; i++)
                tempintVectors[i].Dispose();
            tempintVectors.Clear();

            for (int i = 0; i < tempintMatrices.Length; i++)
                tempintMatrices[i].Dispose();
            tempintMatrices.Clear();
            
            for (int i = 0; i < tempshortVectors.Length; i++)
                tempshortVectors[i].Dispose();
            tempshortVectors.Clear();

            for (int i = 0; i < tempshortMatrices.Length; i++)
                tempshortMatrices[i].Dispose();
            tempshortMatrices.Clear();
            
            for (int i = 0; i < templongVectors.Length; i++)
                templongVectors[i].Dispose();
            templongVectors.Clear();

            for (int i = 0; i < templongMatrices.Length; i++)
                templongMatrices[i].Dispose();
            templongMatrices.Clear();
            

            for (int i = 0; i < TempBoolVectors.Length; i++)
                TempBoolVectors[i].Dispose();
            TempBoolVectors.Clear();

            for (int i = 0; i < TempBoolMatrices.Length; i++)
                TempBoolMatrices[i].Dispose();
            TempBoolMatrices.Clear();
        }

        public void Dispose()
        {
            Clear();

            
            floatVectors.Dispose();
            floatMatrices.Dispose();
            tempfloatMatrices.Dispose();
            tempfloatVectors.Dispose();
            
            doubleVectors.Dispose();
            doubleMatrices.Dispose();
            tempdoubleMatrices.Dispose();
            tempdoubleVectors.Dispose();
            

            
            intVectors.Dispose();
            intMatrices.Dispose();
            tempintMatrices.Dispose();
            tempintVectors.Dispose();
            
            shortVectors.Dispose();
            shortMatrices.Dispose();
            tempshortMatrices.Dispose();
            tempshortVectors.Dispose();
            
            longVectors.Dispose();
            longMatrices.Dispose();
            templongMatrices.Dispose();
            templongVectors.Dispose();
            

            BoolVectors.Dispose();
            BoolMatrices.Dispose();
            TempBoolMatrices.Dispose();
            TempBoolVectors.Dispose();

            Initialized = false;
            Allocator = Allocator.Invalid;
        }
    }
}