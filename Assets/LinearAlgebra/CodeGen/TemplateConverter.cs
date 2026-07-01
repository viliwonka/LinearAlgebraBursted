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

                sourceCode = CopyReplaceAll(sourceCode, relativePath);
                sourceCode = CopyReplaceFill(sourceCode, relativePath);
                sourceCode = CopyReplace(sourceCode, relativePath);
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
                }

                for (int i = 0; i < types.Length; i++) {
                    var typeStr = types[i];
                    var capsTypeStr = capsTypes[i];

                    var targetPath = relativePath.Replace(proxy, typeStr);
                    targetPath = targetPath.Replace(capsProxy, capsTypeStr);

                    //Debug.Log($"TargetPath: {targetPath}");

                    var targetSource = sourceCode.Replace(proxy, typeStr);
                    targetSource = targetSource.Replace(capsProxy, capsTypeStr);
                    targetSource = ChooseReplace(targetSource, i, relativePath);

                    context.AddCode(targetPath, targetSource);
                }
            }

            // singular files, do not multiply
            foreach (var sourceCodePath in singularFilesPaths) {
                var sourceFileName = Path.GetFileName(sourceCodePath);

                if (IgnoreFile(sourceFileName))
                    continue;

                var relativePath = Path.GetRelativePath(sourceFolder, sourceCodePath);

                var targetSource = File.ReadAllText(sourceCodePath);

                targetSource = CopyReplaceAll(targetSource, relativePath);
                targetSource = CopyReplaceFill(targetSource, relativePath);
                targetSource = CopyReplace(targetSource, relativePath);
                targetSource = DeleteThis(targetSource, relativePath);

                var targetPath = relativePath;

                //Debug.Log($"Target: {targetPath}");

                context.AddCode(targetPath, targetSource);
            }
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

        string CopyReplaceFill(string targetSource, string filePathDebug) {

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
                subString = GenerateForAllTypes(subString, false, fill);
                //Debug.Log("After:"+subString);

                targetSource = targetSource.Remove(startIndex, endIndex - startIndex);
                targetSource = targetSource.Insert(startIndex, subString);
                --infinityGuard;
            }

            if (infinityGuard == 0)
                Debug.LogError($"Infinity guard triggered, copyReplace syntax is bad: {filePathDebug}");

            return targetSource;
        }

        string CopyReplace(string targetSource, string filePathDebug) {

            int infinityGuard = 40;

            while (targetSource.Contains(GenUtils.copyMarkerStart) && infinityGuard > 0) {
                int startIndex = targetSource.IndexOf(GenUtils.copyMarkerStart);
                int endIndex = targetSource.IndexOf(GenUtils.copyMarkerEnd, startIndex) + GenUtils.copyMarkerLen;

                int startSubString = startIndex + GenUtils.copyMarkerLen;
                int endSubString = endIndex - GenUtils.copyMarkerLen;

                string subString = targetSource.Substring(startSubString, endSubString - startSubString);

                subString = GenerateForAllTypes(subString);

                targetSource = targetSource.Remove(startIndex, endIndex - startIndex);
                targetSource = targetSource.Insert(startIndex, subString);
                --infinityGuard;
            }

            if (infinityGuard == 0)
                Debug.LogError($"Infinity guard triggered, copyReplace syntax is bad: {filePathDebug}");

            return targetSource;
        }

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
                Debug.LogError($"Infinity guard triggered, copyReplace syntax is bad: {filePathDebug}");

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

        string GenerateForAllTypes(string subString, bool allTypes = false, string fill = "") {

            string result = "";

            string[] types;
            string[] capsTypes;
            string proxy;
            string capsProxy;

            if (allTypes) {
                
                types = GenUtils.floatTypes.Concat(GenUtils.intTypes).Concat(GenUtils.boolTypes).ToArray();
                capsTypes = GenUtils.capsFloatTypes.Concat(GenUtils.capsIntTypes).Concat(GenUtils.boolTypes).ToArray();
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
                // process only int types
                types = GenUtils.intTypes;
                capsTypes = GenUtils.capsIntTypes;
                proxy = GenUtils.iProxy;
                capsProxy = GenUtils.cIProxy;
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