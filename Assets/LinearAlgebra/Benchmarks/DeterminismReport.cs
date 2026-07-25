using System.Collections.Generic;
using System.IO;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using BULA;

using Debug = UnityEngine.Debug;

namespace BULA.Benchmarks
{
    // Hand-written, dtype-agnostic half of the determinism conformance harness: group registry,
    // op/group/root hash folding, and report writing. The per-op Burst jobs and input builders are
    // code-generated per dtype from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/
    // Determinism*.fProxy.cs / Determinism*.iProxy.cs (see those files' headers). Full contract:
    // docs/dev/spec-determinism-conformance-harness.md.
    //
    // Section A (deterministic core: + - * / sqrt only) folds into ROOT; a cross-arch mismatch there
    // is a bug. Section B (native-math-sensitive: DetMath-routed transcendentals) folds into a
    // SEPARATE ROOT-B, so a LINALG_NATIVE_MATH build only perturbs ROOT-B.
    public static class DeterminismReport
    {
        // Bump on ANY change to the case list, case inputs, or fold order — all of these legitimately
        // change every hash below them. Two reports with different revs must never be diffed.
        public const int HarnessRev = 1;

        const uint FoldSeed = 0x9E3779B9u;

        [UnityEditor.MenuItem("Tools/LinearAlgebra/Determinism Report")]
        public static void RunMenuItem() => Run();

        public static void Run()
        {
            var sectionA = new StringBuilder();
            var sectionB = new StringBuilder();
            uint rootA = FoldSeed;
            uint rootB = FoldSeed;

            // ---- Section A: deterministic core, fixed registration order ----
            Group(sectionA, ref rootA, "hash-selftest", HashSelfTestCases());
            Group(sectionA, ref rootA, "blas-dense", Cat(DeterminismDirect.Case_BlasDenseFloat(), DeterminismDirect.Case_BlasDenseDouble()));
            Group(sectionA, ref rootA, "elementwise-core", Cat(DeterminismDirect.Case_ElementwiseCoreFloat(), DeterminismDirect.Case_ElementwiseCoreDouble()));
            Group(sectionA, ref rootA, "norms", Cat(DeterminismDirect.Case_NormsFloat(), DeterminismDirect.Case_NormsDouble()));
            Group(sectionA, ref rootA, "stats-core", Cat(DeterminismStatsMl.Case_StatsCoreFloat(), DeterminismStatsMl.Case_StatsCoreDouble()));
            Group(sectionA, ref rootA, "qr-family", Cat(DeterminismDirect.Case_QrFamilyFloat(), DeterminismDirect.Case_QrFamilyDouble()));
            Group(sectionA, ref rootA, "lu", Cat(DeterminismDirect.Case_LuFloat(), DeterminismDirect.Case_LuDouble()));
            Group(sectionA, ref rootA, "cholesky", Cat(DeterminismDirect.Case_CholeskyFloat(), DeterminismDirect.Case_CholeskyDouble()));
            Group(sectionA, ref rootA, "eigen-sym", Cat(DeterminismEigenSvd.Case_EigenSymFloat(), DeterminismEigenSvd.Case_EigenSymDouble()));
            Group(sectionA, ref rootA, "eigen-nonsym", Cat(DeterminismEigenSvd.Case_EigenNonsymFloat(), DeterminismEigenSvd.Case_EigenNonsymDouble()));
            Group(sectionA, ref rootA, "svd", Cat(DeterminismEigenSvd.Case_SvdFloat(), DeterminismEigenSvd.Case_SvdDouble()));
            Group(sectionA, ref rootA, "fft", Cat(DeterminismStatsMl.Case_FftFloat(), DeterminismStatsMl.Case_FftDouble()));
            Group(sectionA, ref rootA, "krylov-dense", Cat(DeterminismIterativeSparse.Case_KrylovDenseFloat(), DeterminismIterativeSparse.Case_KrylovDenseDouble()));
            Group(sectionA, ref rootA, "sparse-bsr", Cat(DeterminismIterativeSparse.Case_SparseBsrFloat(), DeterminismIterativeSparse.Case_SparseBsrDouble()));
            Group(sectionA, ref rootA, "krylov-sparse-precond", Cat(DeterminismIterativeSparse.Case_KrylovSparsePrecondFloat(), DeterminismIterativeSparse.Case_KrylovSparsePrecondDouble()));
            Group(sectionA, ref rootA, "lobpcg", Cat(DeterminismEigenSvd.Case_LobpcgFloat(), DeterminismEigenSvd.Case_LobpcgDouble()));
            Group(sectionA, ref rootA, "lp-lad", Cat(DeterminismOptimize.Case_LpLadFloat(), DeterminismOptimize.Case_LpLadDouble()));
            Group(sectionA, ref rootA, "qp", Cat(DeterminismOptimize.Case_QpFloat(), DeterminismOptimize.Case_QpDouble()));
            Group(sectionA, ref rootA, "mip", Cat(DeterminismOptimize.Case_MipFloat(), DeterminismOptimize.Case_MipDouble()));
            Group(sectionA, ref rootA, "control", Cat(DeterminismOptimize.Case_ControlFloat(), DeterminismOptimize.Case_ControlDouble()));
            Group(sectionA, ref rootA, "nls-optimize", Cat(DeterminismOptimize.Case_NlsOptimizeFloat(), DeterminismOptimize.Case_NlsOptimizeDouble()));
            Group(sectionA, ref rootA, "ml", Cat(DeterminismStatsMl.Case_MlFloat(), DeterminismStatsMl.Case_MlDouble()));
            Group(sectionA, ref rootA, "histogram-resample-query", Cat(DeterminismStatsMl.Case_HistogramResampleQueryFloat(), DeterminismStatsMl.Case_HistogramResampleQueryDouble()));
            Group(sectionA, ref rootA, "gallery-analysis", Cat(DeterminismStatsMl.Case_GalleryAnalysisFloat(), DeterminismStatsMl.Case_GalleryAnalysisDouble()));
            Group(sectionA, ref rootA, "int-family", Cat(
                DeterminismInt.Case_IntFamilyCoreInt(), DeterminismInt.Case_IntFamilyNormsStatsInt(),
                DeterminismInt.Case_IntFamilyCoreShort(), DeterminismInt.Case_IntFamilyNormsStatsShort(),
                DeterminismInt.Case_IntFamilyCoreLong(), DeterminismInt.Case_IntFamilyNormsStatsLong(),
                DeterminismInt.Case_IntFamilyCoreUInt()));

            // ---- Section B: native-math-sensitive, fixed registration order, folds into ROOT-B only ----
            Group(sectionB, ref rootB, "detmath", Cat(DeterminismNativeSensitive.Case_DetMathFloat(), DeterminismNativeSensitive.Case_DetMathDouble()));
            Group(sectionB, ref rootB, "elementwise-transcendental", Cat(DeterminismNativeSensitive.Case_ElementwiseTranscendentalFloat(), DeterminismNativeSensitive.Case_ElementwiseTranscendentalDouble()));
            Group(sectionB, ref rootB, "random-samplers", Cat(DeterminismNativeSensitive.Case_RandomSamplersFloat(), DeterminismNativeSensitive.Case_RandomSamplersDouble()));
            Group(sectionB, ref rootB, "softmax", Cat(DeterminismNativeSensitive.Case_SoftmaxFloat(), DeterminismNativeSensitive.Case_SoftmaxDouble()));
            Group(sectionB, ref rootB, "dft-signal", Cat(DeterminismNativeSensitive.Case_DftSignalFloat(), DeterminismNativeSensitive.Case_DftSignalDouble()));

            var sb = new StringBuilder();
            sb.Append("=== LinearAlgebra determinism conformance report ===\n");
            sb.Append("rev ").Append(HarnessRev).Append('\n');
            sb.Append("# host: ").Append(UnityEngine.SystemInfo.operatingSystem).Append(" / ").Append(UnityEngine.Application.platform).Append('\n');
            sb.Append("# burst-enabled: ").Append(BurstCompiler.Options.EnableBurstCompilation).Append('\n');
            sb.Append("# detmath-native: ").Append(DetMath.UseNative).Append('\n');
            sb.Append("# dtypes: float double\n");
            sb.Append('\n');
            sb.Append("ROOT ").Append(HexU(rootA)).Append('\n');
            sb.Append("ROOT-B ").Append(HexU(rootB)).Append('\n');
            sb.Append('\n');
            sb.Append(sectionA);
            sb.Append("=== section B: native-math-sensitive (expected to differ across arch under LINALG_NATIVE_MATH) ===\n");
            sb.Append(sectionB);

            Directory.CreateDirectory("TestResults");
            string path = Path.Combine("TestResults", "determinism-report.txt");
            // UTF-8, no BOM, LF-only line endings, so a byte-identical re-run holds exactly (PS 5.1
            // BOM-less UTF-8 trap -- see Tools/*.ps1's own File.ReadAllText convention).
            File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));

            Debug.Log(sb.ToString());
            if (!BurstCompiler.Options.EnableBurstCompilation)
                Debug.Log("Determinism report FAILED: Burst disabled");
            Debug.Log("Determinism report written to " + path);
        }

        // Folds `cases` (already-computed op hashes, in fixed registration order) into one group
        // hash starting from FoldSeed, appends "GROUP <id> <hex>" + one "OP <group>/<case> <hex>"
        // line per case to `sb`, and folds the group hash into `root`.
        static void Group(StringBuilder sb, ref uint root, string name, (string id, uint hash)[] cases)
        {
            uint g = FoldSeed;
            var opLines = new StringBuilder();
            foreach (var (id, h) in cases)
            {
                g = Hash.combine(g, h);
                opLines.Append("OP ").Append(id).Append(' ').Append(HexU(h)).Append('\n');
            }
            root = Hash.combine(root, g);
            sb.Append("GROUP ").Append(name).Append(' ').Append(HexU(g)).Append('\n');
            sb.Append(opLines);
        }

        static (string, uint)[] Cat(params (string, uint)[][] parts)
        {
            var list = new List<(string, uint)>();
            foreach (var p in parts) list.AddRange(p);
            return list.ToArray();
        }

        static string HexU(uint v) => v.ToString("x8");

        // Dtype-agnostic (no fProxy/iProxy dependency): Hash.hash(byte*) over a fixed 0..255 pattern,
        // compared against an independently-computed xxHash32 reference constant (see
        // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/DEVLOG.md for how it was derived) --
        // if the hash kernel itself is broken on a platform, this is the case that says so, emitting
        // the FAIL sentinel 00000000 instead of a plausible-looking wrong hash.
        [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
        struct HashSelfTestKnownAnswerJob : IJob
        {
            public NativeArray<uint> HashOut;

            public unsafe void Execute()
            {
                byte* pattern = stackalloc byte[256];
                for (int i = 0; i < 256; i++) pattern[i] = (byte)i;
                uint h = Hash.hash(pattern, 256, 0);
                HashOut[0] = h;
            }
        }

        static (string id, uint hash)[] HashSelfTestCases()
        {
            const uint knownAnswer = 0x59441253u;

            var hashOut = new NativeArray<uint>(1, Allocator.Persistent);
            var job = new HashSelfTestKnownAnswerJob { HashOut = hashOut };
            job.Run();
            uint computed = hashOut[0];
            hashOut.Dispose();

            uint reported = computed == knownAnswer ? computed : 0x00000000u;

            var floatCases = DeterminismDirect.Case_HashSelfTestFloat();
            var doubleCases = DeterminismDirect.Case_HashSelfTestDouble();

            var result = new List<(string, uint)> { ("hash-selftest/known-answer.bytes", reported) };
            result.AddRange(floatCases);
            result.AddRange(doubleCases);
            return result.ToArray();
        }
    }
}
