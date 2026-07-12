using System.Collections;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.IO;
using System.Linq;

using UnityCodeGen;

using UnityEngine;

namespace LinearAlgebra.CodeGen { 

    public class TemplateConverter {

        public void Execute(GeneratorContext context, string sourceFolder) {

            Debug.Log($"TemplateConverter.Execute(context, sourceFolder: {sourceFolder})");

            string templateRootName = new DirectoryInfo(sourceFolder.TrimEnd('/', '\\')).Name;

            var filesList = Directory.EnumerateFiles(sourceFolder, "*.cs", SearchOption.AllDirectories).ToList();

            List<string> singularFilesPaths = new List<string>();

            // Files that multiply (fProxy -> float, double)
            foreach (var sourceCodePath in filesList) {
                var sourceFileName = Path.GetFileName(sourceCodePath);
                // ignore this file, it's a special case
                if (IgnoreFile(sourceFileName))
                    continue;

                var sourceCode = File.ReadAllText(sourceCodePath);

                if (sourceCode.Contains(GenUtils.singularFileMarker)) {
                    singularFilesPaths.Add(sourceCodePath);
                    continue;
                }

                if (sourceFileName.Contains(GenUtils.fProxy) == false && sourceCode.Contains(GenUtils.fProxy) == false
                    && sourceFileName.Contains(GenUtils.iProxy) == false && sourceCode.Contains(GenUtils.iProxy) == false) {
                    singularFilesPaths.Add(sourceCodePath);
                    continue;
                }

                var relativePath = Path.GetRelativePath(sourceFolder, sourceCodePath);

                // Per-file opt-in (the alsoExpand flag, see GenUtils.cs): widen THIS file's
                // expansion set beyond the default int/short/long rotation (e.g. uint), without
                // touching every other iProxy template. Resolved up front (before CopyReplace runs)
                // so it can ALSO widen any inner iProxy-family copy-replace block in this same file
                // (e.g. a cross-type shortcut block), not just the outer per-file type loop below -
                // both need the identical extra-types list.
                var (extraTypes, extraCaps) = ResolveAlsoExpand(sourceCode, relativePath);
                sourceCode = StripAlsoExpandMarker(sourceCode, relativePath);

                sourceCode = CopyReplaceAll(sourceCode, relativePath);
                sourceCode = CopyReplaceFill(sourceCode, relativePath, extraTypes, extraCaps);
                sourceCode = CopyReplace(sourceCode, relativePath, extraTypes, extraCaps);
                sourceCode = DeleteThis(sourceCode, relativePath);

                string[] types;
                string[] capsTypes;
                string proxy;
                string capsProxy;

                if (sourceFileName.Contains(GenUtils.fProxy)) {

                    types = GenUtils.floatTypes;
                    capsTypes = GenUtils.capsFloatTypes;
                    proxy = GenUtils.fProxy;
                    capsProxy = GenUtils.cFProxy;
                }
                else {

                    types = GenUtils.intTypes;
                    capsTypes = GenUtils.capsIntTypes;
                    proxy = GenUtils.iProxy;
                    capsProxy = GenUtils.cIProxy;

                    if (extraTypes.Length > 0) {
                        types = types.Concat(extraTypes).ToArray();
                        capsTypes = capsTypes.Concat(extraCaps).ToArray();
                    }
                }

                for (int i = 0; i < types.Length; i++) {
                    var typeStr = types[i];
                    var capsTypeStr = capsTypes[i];

                    var targetPath = relativePath.Replace(proxy, typeStr);
                    targetPath = targetPath.Replace(capsProxy, capsTypeStr);

                    //Debug.Log($"TargetPath: {targetPath}");

                    // //+skipFor[...] needs to know typeStr, so - unlike the CopyReplace family
                    // above, which runs once on the shared sourceCode before this loop starts - it
                    // runs HERE, per generated type, on a per-iteration copy of sourceCode (mirrors
                    // ChooseReplace's placement, for the same reason).
                    var perTypeSource = SkipForReplace(sourceCode, typeStr, relativePath);

                    var targetSource = perTypeSource.Replace(proxy, typeStr);
                    targetSource = targetSource.Replace(capsProxy, capsTypeStr);
                    targetSource = ChooseReplace(targetSource, i, relativePath);

                    context.AddCode(targetPath, Banner(templateRootName, relativePath) + targetSource);
                }
            }

            // singular files, do not multiply
            foreach (var sourceCodePath in singularFilesPaths) {
                var sourceFileName = Path.GetFileName(sourceCodePath);

                if (IgnoreFile(sourceFileName))
                    continue;

                var relativePath = Path.GetRelativePath(sourceFolder, sourceCodePath);

                var targetSource = File.ReadAllText(sourceCodePath);

                // A singular file (one merged output, e.g. Arena.cs) can opt into alsoExpand just
                // like a per-type file: its copyReplace/copyReplaceFill blocks widen the same way -
                // see ResolveAlsoExpand/StripAlsoExpandMarker above.
                var (extraTypes, extraCaps) = ResolveAlsoExpand(targetSource, relativePath);
                targetSource = StripAlsoExpandMarker(targetSource, relativePath);

                targetSource = CopyReplaceAll(targetSource, relativePath);
                targetSource = CopyReplaceFill(targetSource, relativePath, extraTypes, extraCaps);
                targetSource = CopyReplace(targetSource, relativePath, extraTypes, extraCaps);
                targetSource = DeleteThis(targetSource, relativePath);

                var targetPath = relativePath;

                //Debug.Log($"Target: {targetPath}");

                context.AddCode(targetPath, Banner(templateRootName, relativePath) + targetSource);
            }
        }

        // Standard <auto-generated> header (the Roslyn convention: IDEs/analyzers treat the file as
        // generated). Names the template so a reader lands on the real source of truth.
        static string Banner(string templateRootName, string relativePath) {
            string p = relativePath.Replace('\\', '/');
            return "// <auto-generated>\n" +
                   $"//   Generated from Assets/LinearAlgebra/CodeGen/{templateRootName}/{p}\n" +
                   "//   DO NOT EDIT BY HAND - edit the template and run Tools/regen.ps1.\n" +
                   "// </auto-generated>\n";
        }

        bool IgnoreFile(string fileName) {
            if (fileName.Contains("proxyStructs")
            || fileName.Contains("markers")
            || fileName.Contains("proxyShims"))
                return true;

            return false;
        }

        void GenerateDirectoryIfItDoesntExist(string basePath, string path) {
            if (Directory.Exists(path) == false)
                Directory.CreateDirectory(path);
        }

        string CopyReplaceFill(string targetSource, string filePathDebug, string[] extraTypes = null, string[] extraCaps = null) {

            int infinityGuard = 40;

            while (targetSource.Contains(GenUtils.copyFillMarkerStart) && infinityGuard > 0) {

                int startIndex = targetSource.IndexOf(GenUtils.copyFillMarkerStart);
                int endIndex = targetSource.IndexOf(GenUtils.copyFillMarkerEnd, startIndex) + GenUtils.copyFillMarkerLen;

                int startSymbolIndex = targetSource.IndexOf("[", startIndex);
                int endSymbolIndex = targetSource.IndexOf("]", startSymbolIndex);

                string fill = targetSource.Substring(startSymbolIndex + 1, endSymbolIndex - startSymbolIndex - 1);
                //Debug.Log("Fill:"+fill);

                int subStringStart = startIndex + GenUtils.copyFillMarkerLen + fill.Length + 2;
                int subStringEnd = endIndex - GenUtils.copyFillMarkerLen;

                string subString = targetSource.Substring(subStringStart, subStringEnd - subStringStart);
                //Debug.Log("Before:"+subString);
                subString = GenerateForAllTypes(subString, false, fill, extraTypes, extraCaps);
                //Debug.Log("After:"+subString);

                targetSource = targetSource.Remove(startIndex, endIndex - startIndex);
                targetSource = targetSource.Insert(startIndex, subString);
                --infinityGuard;
            }

            if (infinityGuard == 0)
                Debug.LogError($"Infinity guard triggered, copyReplaceFill syntax is bad: {filePathDebug}");

            return targetSource;
        }

        string CopyReplace(string targetSource, string filePathDebug, string[] extraTypes = null, string[] extraCaps = null) {

            int infinityGuard = 40;

            while (targetSource.Contains(GenUtils.copyMarkerStart) && infinityGuard > 0) {
                int startIndex = targetSource.IndexOf(GenUtils.copyMarkerStart);
                int endIndex = targetSource.IndexOf(GenUtils.copyMarkerEnd, startIndex) + GenUtils.copyMarkerLen;

                int startSubString = startIndex + GenUtils.copyMarkerLen;
                int endSubString = endIndex - GenUtils.copyMarkerLen;

                string subString = targetSource.Substring(startSubString, endSubString - startSubString);

                subString = GenerateForAllTypes(subString, false, "", extraTypes, extraCaps);

                targetSource = targetSource.Remove(startIndex, endIndex - startIndex);
                targetSource = targetSource.Insert(startIndex, subString);
                --infinityGuard;
            }

            if (infinityGuard == 0)
                Debug.LogError($"Infinity guard triggered, copyReplace syntax is bad: {filePathDebug}");

            return targetSource;
        }

        // NOTE: unlike CopyReplace/CopyReplaceFill, this does NOT thread alsoExpand's extraTypes -
        // GenerateForAllTypes's `allTypes` branch always uses the FIXED
        // float+double+int+short+long+bool list, with no slot for an extra int-family type (where
        // would `uint` sit relative to `bool`?). No current //+copyReplaceAll file (only
        // Pivot.Operations.cs) has an alsoExpand marker, so this is a documented gap, not a bug -
        // if a copyReplaceAll file ever needs uint too, thread extraTypes through here AND decide
        // its position in the merged list first.
        string CopyReplaceAll(string targetSource, string filePathDebug) {

            int infinityGuard = 40;

            while (targetSource.Contains(GenUtils.copyAllMarkerStart) && infinityGuard > 0) {
                int startIndex = targetSource.IndexOf(GenUtils.copyAllMarkerStart);
                int endIndex = targetSource.IndexOf(GenUtils.copyAllMarkerEnd, startIndex) + GenUtils.copyAllMarkerLen;

                int startSubString = startIndex + GenUtils.copyAllMarkerLen;
                int endSubString = endIndex - GenUtils.copyAllMarkerLen;

                string subString = targetSource.Substring(startSubString, endSubString - startSubString);

                subString = GenerateForAllTypes(subString, true);

                targetSource = targetSource.Remove(startIndex, endIndex - startIndex);
                targetSource = targetSource.Insert(startIndex, subString);
                --infinityGuard;
            }

            if (infinityGuard == 0)
                Debug.LogError($"Infinity guard triggered, copyReplaceAll syntax is bad: {filePathDebug}");

            return targetSource;
        }

        string DeleteThis(string targetSource, string filePathDebug) {

            int infinityGuard = 100;

            while (targetSource.Contains(GenUtils.deleteMarkerStart) && infinityGuard > 0) {
                int startIndex = targetSource.IndexOf(GenUtils.deleteMarkerStart);
                int endIndex = targetSource.IndexOf(GenUtils.deleteMarkerEnd) + GenUtils.deleteMarkerLen;

                targetSource = targetSource.Remove(startIndex, endIndex - startIndex);
                --infinityGuard;
            }

            if (infinityGuard == 0)
                Debug.LogError($"Infinity guard triggered, deleteThis syntax is bad: {filePathDebug}");

            return targetSource;
        }

        // Resolves /*+choose[v0|v1|...]*/placeholder/*-choose*/ to v(typeIndex) for the type currently
        // being generated (typeIndex indexes the SAME types[] array the per-type loop in Execute uses,
        // so v0/v1/v2 line up with float/double or int/short/long in that order). Unlike CopyReplace's
        // family this runs ONCE PER GENERATED TYPE (inside the per-type loop, on targetSource), not
        // once on the shared sourceCode before the loop - it needs to know which type is being emitted.
        string ChooseReplace(string targetSource, int typeIndex, string filePathDebug) {

            int infinityGuard = 100;

            while (targetSource.Contains(GenUtils.chooseMarkerStart) && infinityGuard > 0) {

                int startIndex = targetSource.IndexOf(GenUtils.chooseMarkerStart);
                int valuesStart = startIndex + GenUtils.chooseMarkerStart.Length;
                int valuesEnd = targetSource.IndexOf("]", valuesStart);

                if (valuesEnd < 0) {
                    Debug.LogError($"choose marker missing closing ']': {filePathDebug}");
                    break;
                }

                string[] values = targetSource.Substring(valuesStart, valuesEnd - valuesStart).Split('|');

                int markerCommentClose = targetSource.IndexOf("*/", valuesEnd);
                if (markerCommentClose < 0) {
                    Debug.LogError($"choose marker missing closing '*/': {filePathDebug}");
                    break;
                }
                markerCommentClose += 2;

                int endMarkerIndex = targetSource.IndexOf(GenUtils.chooseMarkerEnd, markerCommentClose);
                if (endMarkerIndex < 0) {
                    Debug.LogError($"choose marker missing matching -choose: {filePathDebug}");
                    break;
                }
                int endIndex = endMarkerIndex + GenUtils.chooseMarkerEnd.Length;

                if (typeIndex < 0 || typeIndex >= values.Length) {
                    Debug.LogError($"choose marker has {values.Length} value(s) but type index {typeIndex} was requested: {filePathDebug}");
                    break;
                }

                string chosen = values[typeIndex].Trim();

                targetSource = targetSource.Remove(startIndex, endIndex - startIndex);
                targetSource = targetSource.Insert(startIndex, chosen);
                --infinityGuard;
            }

            if (infinityGuard == 0)
                Debug.LogError($"Infinity guard triggered, choose syntax is bad: {filePathDebug}");

            return targetSource;
        }

        // //alsoExpand[type,...]// - per-file opt-in, see GenUtils.cs. Returns the extra
        // (type, caps) pairs to append to this file's types[]/capsTypes[] arrays; empty arrays if
        // the file doesn't opt in.
        (string[] types, string[] caps) ResolveAlsoExpand(string sourceCode, string filePathDebug) {

            int startIndex = sourceCode.IndexOf(GenUtils.alsoExpandMarkerStart);
            if (startIndex < 0)
                return (System.Array.Empty<string>(), System.Array.Empty<string>());

            // A file may only opt in ONCE - a second marker would otherwise be silently ignored
            // (only the first occurrence is ever parsed below) and ship as unstripped comment text.
            if (sourceCode.IndexOf(GenUtils.alsoExpandMarkerStart, startIndex + GenUtils.alsoExpandMarkerStart.Length) >= 0)
                Debug.LogError($"alsoExpand: a file may only have ONE //alsoExpand[...]// marker (found a second one): {filePathDebug}");

            int listStart = startIndex + GenUtils.alsoExpandMarkerStart.Length;
            int listEnd = sourceCode.IndexOf(GenUtils.alsoExpandMarkerEnd, listStart);
            if (listEnd < 0) {
                Debug.LogError($"alsoExpand marker missing closing ']//': {filePathDebug}");
                return (System.Array.Empty<string>(), System.Array.Empty<string>());
            }

            string[] names = sourceCode.Substring(listStart, listEnd - listStart).Split(',');

            var types = new List<string>();
            var caps = new List<string>();

            foreach (var rawName in names) {
                string name = rawName.Trim();

                // Already part of the file's base int/short/long rotation - listing it again would
                // just duplicate an existing generated type.
                if (System.Array.IndexOf(GenUtils.intTypes, name) >= 0) {
                    Debug.LogError($"alsoExpand: '{name}' is already part of the base int/short/long rotation, listing it again is redundant: {filePathDebug}");
                    continue;
                }

                // Repeated entry within the SAME bracket (e.g. alsoExpand[uint,uint]).
                if (types.Contains(name)) {
                    Debug.LogError($"alsoExpand: duplicate entry '{name}': {filePathDebug}");
                    continue;
                }

                bool found = false;
                foreach (var (type, capsToken) in GenUtils.extraIntTypes) {
                    if (type == name) {
                        types.Add(type);
                        caps.Add(capsToken);
                        found = true;
                        break;
                    }
                }
                if (!found)
                    Debug.LogError($"alsoExpand: unknown type '{name}' (register it in GenUtils.extraIntTypes first): {filePathDebug}");
            }

            return (types.ToArray(), caps.ToArray());
        }

        // Strips the //alsoExpand[...]// marker's own physical line (the marker plus any trailing
        // prose sharing that same line), AND any run of lines immediately following it that are
        // themselves //-only comment lines with no blank line in between - the common case where the
        // marker's line is the FIRST line of a multi-line doc-comment paragraph explaining the flag
        // (see UnsafeMathOP.iProxy.cs/Assume.iProxy.cs/etc.). The run stops at the first line that isn't
        // a bare "//" comment (blank line or real code), which is left untouched - only content
        // CONTIGUOUS with the marker line is ever consumed, so an unrelated standalone doc comment
        // elsewhere in the file (separated by a blank line, or just not adjacent) is never touched.
        // Types were already captured by ResolveAlsoExpand before this runs.
        string StripAlsoExpandMarker(string sourceCode, string filePathDebug) {

            int startIndex = sourceCode.IndexOf(GenUtils.alsoExpandMarkerStart);
            if (startIndex < 0)
                return sourceCode;

            int removeEnd = LineEndInclusive(sourceCode, startIndex);
            while (removeEnd < sourceCode.Length && IsLineCommentLine(sourceCode, removeEnd))
                removeEnd = LineEndInclusive(sourceCode, removeEnd);

            return sourceCode.Remove(startIndex, removeEnd - startIndex);
        }

        // Index right after the newline that ends the line starting at lineStart (or the string's
        // length, if that line has no trailing newline - i.e. it's the last line in the file).
        int LineEndInclusive(string source, int lineStart) {
            int nl = source.IndexOf('\n', lineStart);
            return nl < 0 ? source.Length : nl + 1;
        }

        // True if the line starting at lineStart is a bare line comment (its first non-whitespace
        // characters are "//") - used to find the run of doc-comment lines right after an
        // //alsoExpand[...]// marker.
        bool IsLineCommentLine(string source, int lineStart) {
            int i = lineStart;
            while (i < source.Length && (source[i] == ' ' || source[i] == '\t'))
                i++;
            return i + 1 < source.Length && source[i] == '/' && source[i + 1] == '/';
        }

        // //+skipFor[tag,...] ... //-skipFor : strips the wrapped block from output for any
        // generated type matching a bracket entry (a concrete type name, or the `u` unsigned tag -
        // see GenUtils.cs / SkipForTagMatches). Runs PER GENERATED TYPE, inside the main per-type
        // loop in Execute() (needs typeStr) - mirrors ChooseReplace's placement, unlike CopyReplace's
        // family, which runs once on the shared sourceCode before the per-type loop even starts.
        string SkipForReplace(string sourceCode, string typeStr, string filePathDebug) {

            int infinityGuard = 200;

            while (sourceCode.Contains(GenUtils.skipForMarkerStart) && infinityGuard > 0) {

                int startIndex = sourceCode.IndexOf(GenUtils.skipForMarkerStart);

                int bracketStart = sourceCode.IndexOf('[', startIndex);
                if (bracketStart < 0) {
                    Debug.LogError($"skipFor marker missing '[tags]': {filePathDebug}");
                    break;
                }
                int bracketEnd = sourceCode.IndexOf(']', bracketStart);
                if (bracketEnd < 0) {
                    Debug.LogError($"skipFor marker missing closing ']': {filePathDebug}");
                    break;
                }

                // //+skipFor[...] is a BLOCK marker (like copyReplace's family), not an inline one
                // (like choose) - it must occupy its line alone (only leading whitespace before it,
                // only trailing whitespace after the closing ']'), or the block-removal below would
                // either silently swallow whatever shares its line or wrongly leave it orphaned.
                if (!MarkerAloneOnLine(sourceCode, startIndex, bracketEnd + 1)) {
                    Debug.LogError($"skipFor marker must be alone on its line ('//+skipFor[...]' with only whitespace around it): {filePathDebug}");
                    break;
                }

                int endMarkerIndex = sourceCode.IndexOf(GenUtils.skipForMarkerEnd, bracketEnd);
                if (endMarkerIndex < 0) {
                    Debug.LogError($"skipFor marker missing matching '//-skipFor': {filePathDebug}");
                    break;
                }

                // Reject nesting: a second //+skipFor opening before THIS block's matching
                // //-skipFor would make the naive first-match end-marker search above close on the
                // wrong (inner) marker instead - nesting isn't supported.
                int nestedStart = sourceCode.IndexOf(GenUtils.skipForMarkerStart, startIndex + GenUtils.skipForMarkerStart.Length);
                if (nestedStart >= 0 && nestedStart < endMarkerIndex) {
                    Debug.LogError($"Nested //+skipFor not supported (opened again before the matching '//-skipFor'): {filePathDebug}");
                    break;
                }

                if (!MarkerAloneOnLine(sourceCode, endMarkerIndex, endMarkerIndex + GenUtils.skipForMarkerEnd.Length)) {
                    Debug.LogError($"skipFor marker must be alone on its line ('//-skipFor' with only whitespace around it): {filePathDebug}");
                    break;
                }

                int endIndex = endMarkerIndex + GenUtils.skipForMarkerEnd.Length;

                string[] tags = sourceCode.Substring(bracketStart + 1, bracketEnd - bracketStart - 1).Split(',');

                bool skip = false;
                foreach (var rawTag in tags) {
                    if (SkipForTagMatches(rawTag.Trim(), typeStr)) {
                        skip = true;
                        break;
                    }
                }

                if (skip) {
                    // Tag matched this generated type: drop the marker lines AND the body they wrap.
                    sourceCode = sourceCode.Remove(startIndex, endIndex - startIndex);
                }
                else {
                    // Tag didn't match: keep the body, strip only the marker lines. Remove the END
                    // marker first - it sits after the start marker, so removing it doesn't shift
                    // startIndex/bracketEnd.
                    int startMarkerLen = bracketEnd + 1 - startIndex;
                    sourceCode = sourceCode.Remove(endMarkerIndex, GenUtils.skipForMarkerEnd.Length);
                    sourceCode = sourceCode.Remove(startIndex, startMarkerLen);
                }

                --infinityGuard;
            }

            if (infinityGuard == 0)
                Debug.LogError($"Infinity guard triggered, skipFor syntax is bad: {filePathDebug}");

            // A stray closing marker with no opening //+skipFor[...] would otherwise silently ship
            // as a harmless-looking comment in generated output instead of erroring.
            if (sourceCode.Contains(GenUtils.skipForMarkerEnd))
                Debug.LogError($"Unmatched '//-skipFor' with no opening '//+skipFor[...]': {filePathDebug}");

            return sourceCode;
        }

        bool SkipForTagMatches(string tag, string typeStr) {
            if (tag == "u")
                return System.Array.IndexOf(GenUtils.unsignedTypeNames, typeStr) >= 0;
            return tag == typeStr;
        }

        // True if source[start..end) has nothing but whitespace between it and both of its line's
        // ends (i.e. the token is the only non-whitespace content on its physical line). Used to
        // enforce that //+skipFor[...] / //-skipFor - block markers, like copyReplace's family -
        // occupy their line alone.
        bool MarkerAloneOnLine(string source, int start, int end) {
            int lineStart = start;
            while (lineStart > 0 && source[lineStart - 1] != '\n')
                lineStart--;

            int lineEnd = end;
            while (lineEnd < source.Length && source[lineEnd] != '\n')
                lineEnd++;

            string before = source.Substring(lineStart, start - lineStart);
            string after = source.Substring(end, lineEnd - end);

            return string.IsNullOrWhiteSpace(before) && string.IsNullOrWhiteSpace(after);
        }

        string GenerateForAllTypes(string subString, bool allTypes = false, string fill = "", string[] extraTypes = null, string[] extraCaps = null) {

            string result = "";

            string[] types;
            string[] capsTypes;
            string proxy;
            string capsProxy;

            if (allTypes) {

                types = GenUtils.floatTypes.Concat(GenUtils.intTypes).Concat(GenUtils.boolTypes).ToArray();
                capsTypes = GenUtils.capsFloatTypes.Concat(GenUtils.capsIntTypes).Concat(GenUtils.capsBoolTypes).ToArray();
                proxy = GenUtils.fProxy; //fProxy is used as a proxy for all types, including int :)
                capsProxy = GenUtils.cFProxy;
            }
            else if (subString.Contains(GenUtils.fProxy)) {
                // process only float types
                types = GenUtils.floatTypes;
                capsTypes = GenUtils.capsFloatTypes;
                proxy = GenUtils.fProxy;
                capsProxy = GenUtils.cFProxy;
            }
            else {
                // process only int types (plus this file's alsoExpand extras, if any - e.g. uint)
                types = GenUtils.intTypes;
                capsTypes = GenUtils.capsIntTypes;
                proxy = GenUtils.iProxy;
                capsProxy = GenUtils.cIProxy;

                if (extraTypes != null && extraTypes.Length > 0) {
                    types = types.Concat(extraTypes).ToArray();
                    capsTypes = capsTypes.Concat(extraCaps).ToArray();
                }
            }

            for (int i = 0; i < types.Length; i++) {
                var typeStr = types[i];
                result += subString.Replace(proxy, typeStr).Replace(capsProxy, capsTypes[i]);

                if (string.IsNullOrEmpty(fill) == false && i != types.Length - 1)
                    result += fill;
                //result += "\n";
            }

            return result;
        }
    }

}