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


        public const string sourceTemplateFolder = "Assets/LinearAlgebra/CodeGen/TemplateSource/";
        public const string sourceTestsTemplateFolder = "Assets/LinearAlgebra/CodeGen/TemplateSourceTests/";

        public const string generatedFolder = "Assets/LinearAlgebra/Source/Generated/";
        public const string generatedTestsFolder = "Assets/LinearAlgebra/SourceTests/Generated/";

        public const string shortcutsTemplateFile = "Assets/LinearAlgebra/CodeGen/Templates/shortcuts.template.txt";
        public const string constructorsTemplateFile = "Assets/LinearAlgebra/CodeGen/Templates/constructors.template.txt";

        public const string copyMarkerStart = "//+copyReplace";
        public const string copyMarkerEnd   = "//-copyReplace";
        public static int copyMarkerLen = copyMarkerStart.Length;

        // similar to copyReplace, but it's also filling inbetween copies
        // syntax: "//+copyReplaceFill[+]"
        public const string copyFillMarkerStart = "//+copyReplaceFill";
        public const string copyFillMarkerEnd = "//-copyReplaceFill";
        public static int copyFillMarkerLen = copyFillMarkerStart.Length; 

        public const string deleteMarkerStart = "//+deleteThis";
        public const string deleteMarkerEnd = "//-deleteThis";
        public static int deleteMarkerLen = deleteMarkerStart.Length;

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