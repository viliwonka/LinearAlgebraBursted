using System.Runtime.CompilerServices;

// Lets the hand-authored test assembly (BurstLinearAlgebra.Tests) see this assembly's internal types.
[assembly: InternalsVisibleTo("BurstLinearAlgebra.Tests")]
// The template-side twin of the grant above: the template test assembly compiles the same test
// sources against THIS assembly (e.g. ChooseMarkerTests exercising the internal ChooseMarkerDemo).
[assembly: InternalsVisibleTo("BurstLinearAlgebra.TemplateSource.Tests-firstpass")]
