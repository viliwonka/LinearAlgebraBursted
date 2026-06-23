<#
  Shared helpers for headless Unity runs. Dot-source this; do not run it directly.
    . "$PSScriptRoot\_unity-common.ps1"
#>

function Get-ProjectRoot {
  return (Resolve-Path "$PSScriptRoot\..").Path
}

function Get-UnityPath {
  # $env:UNITY_PATH overrides; otherwise match ProjectSettings/ProjectVersion.txt
  # against the installed Hub editors, falling back to the newest installed.
  if ($env:UNITY_PATH) {
    if (-not (Test-Path $env:UNITY_PATH)) { throw "UNITY_PATH set but not found: $env:UNITY_PATH" }
    return $env:UNITY_PATH
  }
  $versionFile = Join-Path (Get-ProjectRoot) "ProjectSettings\ProjectVersion.txt"
  $version = $null
  if (Test-Path $versionFile) {
    $line = Select-String -Path $versionFile -Pattern "^m_EditorVersion:\s*(.+)$" | Select-Object -First 1
    if ($line) { $version = $line.Matches[0].Groups[1].Value.Trim() }
  }
  $hub = "C:\Program Files\Unity\Hub\Editor"
  if ($version) {
    $exe = Join-Path $hub "$version\Editor\Unity.exe"
    if (Test-Path $exe) { return $exe }
    Write-Host "WARN: project wants Unity $version but it is not installed under $hub"
  }
  $newest = Get-ChildItem $hub -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "Editor\Unity.exe" } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1
  if ($newest) { return $newest }
  throw "Could not find Unity.exe. Set UNITY_PATH to the editor matching $version."
}

function Clear-StaleLock {
  # A lockfile left by an editor that didn't shut down cleanly makes batchmode
  # abort instantly. Remove it only when no Unity.exe is actually running.
  $lock = Join-Path (Get-ProjectRoot) "Temp\UnityLockfile"
  if (-not (Test-Path $lock)) { return }
  if (Get-Process Unity -ErrorAction SilentlyContinue) {
    throw "Unity Editor is running with this project open. Close it before running headless."
  }
  try {
    $fs = [System.IO.File]::Open($lock, 'Open', 'ReadWrite', 'None'); $fs.Close()
    Remove-Item $lock -Force
    Write-Host "Removed stale Unity lockfile."
  } catch {
    throw "Project lockfile is held by a live process. Close Unity and retry."
  }
}

function Invoke-Unity {
  # Launches Unity in batchmode and BLOCKS until it exits. Uses Start-Process
  # -Wait because the call operator (&) does not reliably wait on Unity.exe
  # (a Windows GUI-subsystem app) and can race ahead of file writes.
  param(
    [Parameter(Mandatory)][string[]]$Arguments,
    [Parameter(Mandatory)][string]$LogFile
  )
  $unity = Get-UnityPath
  Clear-StaleLock
  $root = Get-ProjectRoot
  Write-Host "Unity:   $unity"
  Write-Host "Project: $root"
  New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null
  $all = @("-batchmode", "-projectPath", $root, "-logFile", $LogFile) + $Arguments
  $proc = Start-Process -FilePath $unity -ArgumentList $all -PassThru -Wait
  return [int]$proc.ExitCode
}

function Get-CompileErrors {
  # Unity logs C# compile errors as:  Path/File.cs(12,34): error CS1002: ; expected
  param([Parameter(Mandatory)][string]$LogFile)
  if (-not (Test-Path $LogFile)) { return @() }
  return @(Select-String -Path $LogFile -Pattern "error CS\d+" |
           ForEach-Object { $_.Line.Trim() } |
           Select-Object -Unique)
}
