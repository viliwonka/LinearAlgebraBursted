// Stand-in for the slice of the UnityCodeGen package (Packages/com.annulusgames.unity-codegen@.../
// Editor/Core/{CodeText,GeneratorContext}.cs) that TemplateConverter.cs depends on. Mirrors those
// two types field-for-field and method-for-method so TemplateConverter.cs compiles and behaves
// unmodified; the only difference is codeList/overrideFolderPath are public here (the real package
// marks them `internal`, which only ScriptFileGenerator - also inside the package - needs to read)
// so this project's own writer (Program.WriteScriptsFromContext, replicating
// ScriptFileGenerator.GenerateScriptFromContext) can read them too.
namespace UnityCodeGen
{
    public class CodeText
    {
        public string fileName;
        public string text;
    }

    public sealed class GeneratorContext
    {
        private readonly System.Collections.Generic.List<CodeText> _codeList = new System.Collections.Generic.List<CodeText>();
        public System.Collections.Generic.IReadOnlyList<CodeText> codeList => _codeList;

        public string overrideFolderPath { get; private set; }

        public void AddCode(string fileName, string text)
        {
            _codeList.Add(new CodeText { fileName = fileName, text = text });
        }

        public void OverrideFolderPath(string path)
        {
            overrideFolderPath = path;
        }
    }
}
