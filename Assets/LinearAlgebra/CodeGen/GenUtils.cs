using System.IO;
using UnityCodeGen;
using UnityEngine;
using System.Numerics;
namespace LinearAlgebra.CodeGen
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
    */
    public static class GenUtils
    {
        public const string fProxy = nameof(LinearAlgebra.fProxy);
        public const string iProxy = nameof(LinearAlgebra.iProxy);

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

        public const string generatedFolder = "Assets/LinearAlgebra/Source/Generated/";
        public const string generatedTestsFolder = "Assets/LinearAlgebra/SourceTests/Generated/";

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