using System.Collections.Generic;
using System.IO;
using System.Linq;

using UnityCodeGen;
using UnityEngine;

namespace LinearAlgebra.CodeGen
{
    [Generator]
    public class TemplateSourceGenerator : ICodeGenerator // Inherits ICodeGenerator
    {
        public void Execute(GeneratorContext context) {

            TemplateConverter converter = new TemplateConverter();

            string targetBasePath = GenUtils.generatedFolder;
            context.OverrideFolderPath(targetBasePath);

            converter.Execute(context, GenUtils.sourceTemplateFolder);
        }

    }
}