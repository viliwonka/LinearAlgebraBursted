<#
.SYNOPSIS
  Runs the project's template-to-source code generation headlessly.

.DESCRIPTION
  Default: drives the converter directly via a small .NET console host
  (Tools/CodegenBootstrap) that compiles the REAL generator files
  (Assets/LinearAlgebra/CodeGen/{GenUtils,TemplateConverter}.cs - not copies)
  and runs them the same way the project's three [Generator] wrappers do
  (TemplateSource -> Source, TemplateSourceTests -> SourceTests/
  Generated, TemplateSourceBenchmarks -> Benchmarks/Generated), writing
  files with the same logic UnityCodeGen's
  ScriptFileGenerator uses (skip-if-byte-identical, UTF-8 no BOM). Crucially
  this does NOT require the project to already compile, so it can regenerate
  correctly even right after a generated type was renamed/moved/deleted and
  the tree is currently broken - no more hand-editing generated files to
  unstick codegen. See Tools/CodegenBootstrap/Program.cs.

  After generating, reports which files changed (git) so you can see the diff.

.PARAMETER Check
  Drift check: exit 1 if generation produced any changes (i.e. committed generated
  files are out of sync with the templates). Useful for CI / pre-commit.

.PARAMETER Unity
  Fall back to the OLD path: Unity's public entry point
  (UnityCodeGen.UnityCodeGenUtility.Generate) via -executeMethod. This discovers
  every [Generator] via reflection, which requires the project to already
  compile - so it CANNOT recover from the deadlock described above. Kept only
  as a cross-check against the headless path; close the Unity Editor first.
#>
param([switch]$Check, [switch]$Unity)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_unity-common.ps1"

$root = Get-ProjectRoot
$Log  = Join-Path $root "TestResults\codegen.log"

# Prune generated files whose source template was renamed/moved/deleted since the last
# run - these orphans would otherwise sit alongside fresh output and either duplicate a
# class (CS0111) or block compilation outright. Pure file-system logic (no Unity, no
# compile needed), so it's safe to run even when the tree currently doesn't compile.
# See prune-orphaned-generated.ps1.
& "$PSScriptRoot\prune-orphaned-generated.ps1"

if ($Unity) {
  Write-Host "Running codegen (UnityCodeGen.UnityCodeGenUtility.Generate, -Unity fallback)..."
  $exit = Invoke-Unity `
    -Arguments @("-nographics", "-quit", "-executeMethod", "UnityCodeGen.UnityCodeGenUtility.Generate") `
    -LogFile $Log
  Write-Host "Unity exit code: $exit"

  $compileErrors = Get-CompileErrors $Log
  if ($compileErrors.Count -gt 0) {
    Write-Host "`nFAIL: project does not compile, so codegen could not run:"
    $compileErrors | ForEach-Object { Write-Host "  $_" }
    exit 1
  }
  # -executeMethod silently does nothing if the type/method can't be resolved.
  if (Select-String -Path $Log -Pattern "executeMethod method.*could not|couldn't be found" -Quiet) {
    Write-Host "`nFAIL: Unity could not find UnityCodeGenUtility.Generate. See $Log"
    exit 1
  }
  if ($exit -ne 0) {
    Write-Host "`nFAIL: Unity exited $exit. Last 40 log lines:"
    Get-Content $Log -Tail 40
    exit 1
  }
} else {
  Write-Host "Running codegen (headless CodegenBootstrap, no Unity required)..."
  $bootstrapProj = Join-Path $PSScriptRoot "CodegenBootstrap\CodegenBootstrap.csproj"
  & dotnet build $bootstrapProj -c Release --nologo -v quiet
  if ($LASTEXITCODE -ne 0) {
    Write-Host "`nFAIL: Tools/CodegenBootstrap failed to build."
    exit 1
  }
  $bootstrapDll = Join-Path $PSScriptRoot "CodegenBootstrap\bin\Release\net10.0\CodegenBootstrap.dll"
  & dotnet $bootstrapDll $root
  $exit = $LASTEXITCODE
  Write-Host "CodegenBootstrap exit code: $exit"
  if ($exit -ne 0) {
    Write-Host "`nFAIL: CodegenBootstrap did not complete successfully."
    exit 1
  }
}

# Report what changed under version control.
Push-Location $root
try {
  # Codegen OUTPUT trees only (CodegenBootstrap writes Source, SourceTests/Generated,
  # Benchmarks/Generated). The CodeGen template tree is INPUT: uncommitted template edits there
  # are not generated-file drift and must not fail -Check.
  $changed = @(git status --porcelain -- "Assets/LinearAlgebra/Source" "Assets/LinearAlgebra/SourceTests/Generated" "Assets/LinearAlgebra/Benchmarks/Generated" 2>$null)
} finally { Pop-Location }

if ($changed.Count -eq 0) {
  Write-Host "Codegen complete. No changes - generated files are in sync with the templates."
  exit 0
}

Write-Host "`nCodegen changed $($changed.Count) file(s):"
$changed | ForEach-Object { Write-Host "  $_" }
if ($Check) {
  Write-Host "`nFAIL (-Check): generated files were out of sync with templates. Commit the regenerated files."
  exit 1
}
exit 0
