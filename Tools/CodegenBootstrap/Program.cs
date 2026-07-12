using System;
using System.Collections.Generic;
using System.IO;

using LinearAlgebra.CodeGen;
using UnityCodeGen;

namespace CodegenBootstrap
{
    // Headless equivalent of Unity's "-executeMethod UnityCodeGen.UnityCodeGenUtility.Generate":
    // runs the project's three [Generator] wrappers (TemplateSourceGenerator,
    // TemplateSourceTestsGenerator, TemplateSourceBenchmarksGenerator) without needing Unity
    // (and therefore without needing the project to already compile). Each wrapper is just
    //   context.OverrideFolderPath(<outputBase>); new TemplateConverter().Execute(context, <sourceFolder>);
    // so that's reproduced directly here (folders taken straight from GenUtils, the same constants
    // the wrappers use), and the codeList it produces is written to disk by WriteScriptsFromContext,
    // a line-for-line port of ScriptFileGenerator.GenerateScriptFromContext (see
    // Packages/com.annulusgames.unity-codegen@.../Editor/Utils/ScriptFileGenerator.cs) - same path
    // joining, same "skip if identical" check, same File.WriteAllText (UTF-8, no BOM) semantics, so
    // its output is byte-for-byte identical to what a Unity-hosted run would have written.
    internal static class Program
    {
        private static int Main(string[] args)
        {
            string repoRoot;
            try
            {
                repoRoot = args.Length > 0 ? Path.GetFullPath(args[0]) : ResolveRepoRoot();
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"CodegenBootstrap: {ex.Message}");
                return 1;
            }

            var pairs = new (string source, string output)[]
            {
                (
                    Path.Combine(repoRoot, GenUtils.sourceTemplateFolder),
                    Path.Combine(repoRoot, GenUtils.generatedFolder)
                ),
                (
                    Path.Combine(repoRoot, GenUtils.sourceTestsTemplateFolder),
                    Path.Combine(repoRoot, GenUtils.generatedTestsFolder)
                ),
                (
                    Path.Combine(repoRoot, GenUtils.sourceBenchmarksTemplateFolder),
                    Path.Combine(repoRoot, GenUtils.generatedBenchmarksFolder)
                ),
            };

            bool anyChanged = false;
            foreach (var (source, output) in pairs)
            {
                if (!Directory.Exists(source))
                {
                    Console.Error.WriteLine($"CodegenBootstrap: source folder not found: {source}");
                    return 1;
                }

                Console.WriteLine($"CodegenBootstrap: {source} -> {output}");

                var context = new GeneratorContext();
                context.OverrideFolderPath(output);
                new TemplateConverter().Execute(context, source);

                if (WriteScriptsFromContext(context))
                {
                    anyChanged = true;
                }
            }

            Console.WriteLine(anyChanged
                ? "CodegenBootstrap: done, files changed."
                : "CodegenBootstrap: done, no changes (generated output already up to date).");
            return 0;
        }

        // Walks up from the executable's location until it finds the repo root (identified by the
        // real generator source file living where GenUtils expects it), so the bootstrap can be run
        // from anywhere without an explicit argument.
        private static string ResolveRepoRoot()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null)
            {
                var marker = Path.Combine(dir.FullName, "Assets", "LinearAlgebra", "CodeGen", "GenUtils.cs");
                if (File.Exists(marker))
                    return dir.FullName;
                dir = dir.Parent;
            }

            throw new InvalidOperationException(
                "Could not resolve the repo root by walking up from the executable location; " +
                "pass it explicitly as the first argument.");
        }

        // Port of UnityCodeGen's Editor/Utils/ScriptFileGenerator.GenerateScriptFromContext: same
        // path-splitting/joining, same "leave the file alone if its contents already match" skip,
        // same unconditional File.WriteAllText for anything new/changed. Does NOT delete files that
        // are no longer produced (neither does the original) - that's prune-orphaned-generated.ps1's
        // job, run separately by regen.ps1 before this.
        private static bool WriteScriptsFromContext(GeneratorContext context)
        {
            bool changed = false;

            string folderPath = context.overrideFolderPath;
            if (folderPath == null)
                throw new InvalidOperationException("GeneratorContext.OverrideFolderPath was never called.");

            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            foreach (CodeText code in context.codeList)
            {
                string[] hierarchy = code.fileName.Split('/', '\\');
                string path = folderPath;
                for (int i = 0; i < hierarchy.Length; i++)
                {
                    path += "/" + hierarchy[i];
                    if (i == hierarchy.Length - 1) break;
                    if (!Directory.Exists(path))
                    {
                        Directory.CreateDirectory(path);
                    }
                }

                if (File.Exists(path))
                {
                    string text = File.ReadAllText(path);
                    if (text == code.text)
                    {
                        continue;
                    }
                }

                // .NET's File.WriteAllText(path, contents) defaults to UTF-8 without a byte-order
                // mark, matching the real ScriptFileGenerator's File.WriteAllText(path, code.text)
                // call exactly (same overload, same default encoding).
                File.WriteAllText(path, code.text);
                changed = true;
            }

            return changed;
        }
    }
}
