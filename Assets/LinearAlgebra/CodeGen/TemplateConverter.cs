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
                subString = GenerateForAllFloatTypes(subString, fill);
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

                subString = GenerateForAllFloatTypes(subString);

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

        string GenerateForAllFloatTypes(string subString, string fill = "") {
            string result = "";


            string[] types;
            string[] capsTypes;
            string proxy;
            string capsProxy;
            if (subString.Contains(GenUtils.fProxy)) {

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
                result += subString.Replace(proxy, typeStr).Replace(capsProxy, capsTypes[i]);

                if (string.IsNullOrEmpty(fill) == false && i != types.Length - 1)
                    result += fill;
                //result += "\n";
            }

            return result;
        }
    }

}