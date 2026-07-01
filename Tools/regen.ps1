<#
.SYNOPSIS
  Runs the project's template-to-source code generation headlessly.

.DESCRIPTION
  Invokes UnityCodeGen's public entry point (UnityCodeGenUtility.Generate) via
  -executeMethod. That discovers every [Generator] in the project (TemplateSource,
  TemplateSourceTests, Constructors, Shortcuts), expands the templates and writes
  the generated files, then refreshes the AssetDatabase.

  Requires the project to currently compile (so the generator assembly can load).
  After generating, reports which files changed (git) so you can see the diff.

  NOTE: Unity locks the project, so close the Unity Editor before running this.

.PARAMETER Check
  Drift check: exit 1 if generation produced any changes (i.e. committed generated
  files are out of sync with the templates). Useful for CI / pre-commit.
#>
param([switch]$Check)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_unity-common.ps1"

$root = Get-ProjectRoot
$Log  = Join-Path $root "TestResults\codegen.log"

# Prune generated files whose source template was renamed/moved/deleted since the last
# run - these orphans would otherwise sit alongside fresh output and either duplicate a
# class (CS0111) or block compilation outright, which then blocks codegen from running
# at all (it needs the project to compile first). See prune-orphaned-generated.ps1.
& "$PSScriptRoot\prune-orphaned-generated.ps1"

Write-Host "Running codegen (UnityCodeGen.UnityCodeGenUtility.Generate)..."
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

# Report what changed under version control.
Push-Location $root
try {
  $changed = @(git status --porcelain -- "Assets/LinearAlgebra/Source" "Assets/LinearAlgebra/CodeGen" 2>$null)
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
