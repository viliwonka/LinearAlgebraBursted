using System.Collections.Generic;
using System.IO;
using System.Linq;

using UnityCodeGen;
using UnityEngine;

namespace LinearAlgebra.CodeGen
{
    [Generator]
    public class TemplateSourceBenchmarksGenerator : ICodeGenerator // Inherits ICodeGenerator
    {
        public void Execute(GeneratorContext context) {

            TemplateConverter converter = new TemplateConverter();

            string targetBasePath = GenUtils.generatedBenchmarksFolder;
            context.OverrideFolderPath(targetBasePath);

            converter.Execute(context, GenUtils.sourceBenchmarksTemplateFolder);
        }

    }
}
