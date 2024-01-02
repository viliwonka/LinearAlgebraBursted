using System.IO;
using System.Security.Principal;

using UnityCodeGen;
using UnityEngine;

namespace LinearAlgebra.CodeGen
{
    //[Generator]
    public class ShortcutGenerator : ICodeGenerator // Inherits ICodeGenerator
    {

        public void Execute(GeneratorContext context) // Implement Execute method
        {
            Debug.Log("ShortcutGenerator.Execute()");
            context.OverrideFolderPath(GenUtils.generatedFolder + "structs");
            
            string template = File.ReadAllText(GenUtils.shortcutsTemplateFile);

            Debug.Log(template);
            foreach (var bType in GenUtils.baseType)
            {
                foreach (var mType in GenUtils.matTypes)
                {
                    var concreteType = bType + mType;
                    var code = template.Replace("<cType>", concreteType);

                    context.AddCode($@"{concreteType}.shortcuts.gen.cs", code);
                }
            }
        }

    }
}