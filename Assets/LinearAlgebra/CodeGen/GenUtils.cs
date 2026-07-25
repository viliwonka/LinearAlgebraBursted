using System.IO;
using UnityCodeGen;
using UnityEngine;
using System.Numerics;
namespace BULA.CodeGen
{
    /*
        Template source code is also compilable, but it's not meant to be used directly.
        It uses proxy structs where the actual float or double, e.g. floatN, floatMxN, doubleN, doubleMxN.
        fProxy is floating point proxy.
        iProxy is integer proxy.
        This file contains all the utility functions used by the code generator.
        It also kind of describes code gen syntax

        //(+/-)copyReplace are markers for code generator to replace the code with concrete types
        //(+/-)copyReplaceFill[symbol] are markers for code generator to replace the code with concrete types and fills inbetween with symbols
        //(+/-)deleteThis are markers for code generator to delete the code segment
        //+choose[v0|v1|...] ... -choose are INLINE markers, block-commented (not line-commented like
        //the markers above) so they can sit mid-statement: the wrapped placeholder value is replaced
        //with v0 for the first generated type, v1 for the second, etc. (float|double for an fProxy
        //file; int|short|long for an iProxy file). Lets a magic-number constant differ per generated
        //type - see ChooseMarkerDemo.fProxy.cs / ChooseMarkerDemo.iProxy.cs for worked examples.
        //+skipFor[tag,...] ... -skipFor are BLOCK markers (line-commented, like copyReplace/deleteThis
        //above - unlike choose, this wraps a body, so it doesn't need to sit mid-statement): the
        //wrapped block is OMITTED from output for any generated type matching a bracket entry (the
        //marker lines themselves never appear in any output, matched or not). Entries are either a
        //concrete type name ("uint", "short") or the `u` tag (matches any UNSIGNED concrete type).
        //Used to wrap signed-only code (unary minus, negative literals) that would fail to compile
        //once an unsigned type joins an iProxy file's rotation.
        //+emitFor[tag,...] ... -emitFor are BLOCK markers, the INVERSE of skipFor, for per-type
        //bodies that CANNOT compile inside the template assembly (e.g. intrinsics that type-check
        //for only one generated type, or an alternate body whose locals would collide with the
        //primary body's): every body line is hidden behind a leading '//!' so the template
        //compiles only the primary body; for a generated type matching a bracket entry the block
        //is emitted with the '//!' prefixes stripped, for every other type the whole block is
        //dropped. Same tag grammar and block rules as skipFor (markers alone on their lines, no
        //nesting). Pair it with a skipFor on the primary body: skipFor[X] primary + emitFor[X]
        //alternate gives type X the alternate and everyone else the primary.
        //alsoExpand[type,...]// is a per-FILE opt-in FLAG (single line, no closing marker - mirrors
        //singularFile// below): it appends the listed concrete type(s) - which must be pre-registered
        //in extraIntTypes - to THIS iProxy file's normal int/short/long expansion set, without
        //affecting any other iProxy template. This is how a new int-family type (uint) lights up on
        //only the handful of files that are ready for it instead of every iProxy file at once.
    */
    public static class GenUtils
    {
        public const string fProxy = nameof(BULA.fProxy);
        public const string iProxy = nameof(BULA.iProxy);

        public const string cFProxy = "FProxy";
        public const string cIProxy = "IProxy";

        public static string[] floatTypes = new[] { "float", "double" };
        public static string[] capsFloatTypes = new[] { "Float", "Double" };

        public static string[] intTypes = new[] { "int", "short", "long" };
        public static string[] capsIntTypes = new[] { "Int", "Short", "Long" };

        public static string[] boolTypes = new[] { "bool" };
        public static string[] capsBoolTypes = new[] { "Bool" };

        public const string sourceTemplateFolder = "Assets/LinearAlgebra/CodeGen/TemplateSource/";
        public const string sourceTestsTemplateFolder = "Assets/LinearAlgebra/CodeGen/TemplateSourceTests/";
        public const string sourceBenchmarksTemplateFolder = "Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/";

        public const string generatedFolder = "Assets/LinearAlgebra/Source/";  // package root IS the generated tree (only package.json + asmdef are hand-placed)
        public const string generatedTestsFolder = "Assets/LinearAlgebra/SourceTests/Generated/";
        public const string generatedBenchmarksFolder = "Assets/LinearAlgebra/Benchmarks/Generated/";

        public const string copyMarkerStart = "//+copyReplace";
        public const string copyMarkerEnd   = "//-copyReplace";
        public static int copyMarkerLen = copyMarkerStart.Length;

        public const string copyAllMarkerStart = "//+copyReplaceAll";
        public const string copyAllMarkerEnd = "//-copyReplaceAll";
        public static int copyAllMarkerLen = copyAllMarkerStart.Length;

        // similar to copyReplace, but it's also filling inbetween copies
        // syntax: "//+copyReplaceFill[+]"
        public const string copyFillMarkerStart = "//+copyReplaceFill";
        public const string copyFillMarkerEnd = "//-copyReplaceFill";
        public static int copyFillMarkerLen = copyFillMarkerStart.Length; 

        public const string deleteMarkerStart = "//+deleteThis";
        public const string deleteMarkerEnd = "//-deleteThis";
        public static int deleteMarkerLen = deleteMarkerStart.Length;

        // Inline per-generated-type literal substitution: /*+choose[v0|v1|...]*/placeholder/*-choose*/
        // resolves to v(typeIndex) - v0 for the first type in the file's types[] array, v1 for the
        // second, etc. Block comments (not // line comments) so the marker can sit mid-statement.
        public const string chooseMarkerStart = "/*+choose[";
        public const string chooseMarkerEnd = "/*-choose*/";

        // //+skipFor[tag,...] ... //-skipFor : strips the wrapped block from output for any
        // generated type matching a bracket entry. See the doc comment above and
        // TemplateConverter.SkipForReplace/SkipForTagMatches.
        public const string skipForMarkerStart = "//+skipFor";
        public const string skipForMarkerEnd = "//-skipFor";

        public const string emitForMarkerStart = "//+emitFor";
        public const string emitForMarkerEnd = "//-emitFor";
        public const string emitForLinePrefix = "//!";

        // Concrete type names the `u` tag in a //+skipFor[...] bracket matches. Add "ushort"/"byte"
        // here (and to extraIntTypes below) when those land.
        public static readonly string[] unsignedTypeNames = { "uint" };

        // //alsoExpand[type,...]// : per-file opt-in flag, see the doc comment above and
        // TemplateConverter.ResolveAlsoExpand. Not bracket-block-closed like copyReplace/skipFor -
        // it's a single-line FLAG like singularFileMarker below, just with a payload.
        public const string alsoExpandMarkerStart = "//alsoExpand[";
        public const string alsoExpandMarkerEnd = "]//";

        // (concrete type, PascalCase caps token) pairs addressable via //alsoExpand[...]. The caps
        // token mirrors capsIntTypes' shape even though no current template actually uses the caps
        // "IProxy" token.
        public static readonly (string type, string caps)[] extraIntTypes = new[] {
            ("uint", "UInt"),
        };

        // tells compiler that file is singular and should not be copied for each type
        public const string singularFileMarker = "//singularFile//";

        public readonly static string[] baseType = {
            "float",
            "bool",
        };

        public readonly static string[] matTypes = {
            "N",
            "MxN"
        };

    }
}