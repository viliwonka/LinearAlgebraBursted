// GenUtils.cs anchors its "fProxy"/"iProxy" string constants to real type names via
// nameof(BULA.fProxy) / nameof(BULA.iProxy) (see
// Assets/LinearAlgebra/CodeGen/GenUtils.cs) so that renaming those marker types is a compile error
// here too, not a silently-stale string literal. The real types live in
// Assets/LinearAlgebra/CodeGen/TemplateSource/proxyStructs.cs with a full arithmetic/comparison
// operator surface plus a Unity.Burst attribute - none of which TemplateConverter.cs ever touches
// at runtime (that file is a compile-time safety net for templates, not generator logic). Only the
// type *names* matter to nameof(), so this stub keeps them as bare marker types rather than pulling
// proxyStructs.cs (and a Unity.Burst stub) into this project.
namespace BULA
{
    internal struct fProxy { }
    internal struct iProxy { }
}
