using System.IO;
using System.Security.Principal;

using UnityCodeGen;
using UnityEngine;

namespace LinearAlgebra.CodeGen
{
    //[Generator]
    public class ConstructorsGenerator : ICodeGenerator // Inherits ICodeGenerator
    {
        public void Execute(GeneratorContext context) // Implement Execute method
        {
            Debug.Log($"{nameof(ConstructorsGenerator)}.Execute()");

            context.OverrideFolderPath(GenUtils.generatedFolder + "structs");
            
            string template = File.ReadAllText(GenUtils.constructorsTemplateFile);

            Debug.Log(template);
            foreach (var bType in GenUtils.baseType)
            {
                Debug.Log($"baseType: {bType}");

                foreach (var mType in GenUtils.matTypes)
                {
                    var concreteType = bType + mType;
                    var code = template
                        .Replace("<ctype>", concreteType)
                        .Replace("<type>", bType);

                    context.AddCode($@"{concreteType}.constructors.gen.cs", code);
                }
            }
        }

    }
}