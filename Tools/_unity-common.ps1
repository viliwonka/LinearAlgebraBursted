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
  # Launches Unity in batchmode and BLOCKS until it exits. We start WITHOUT
  # -Wait (so we can optionally pin CPU affinity on the live process) and then
  # WaitForExit() ourselves -- the call operator (&) does not reliably wait on
  # Unity.exe (a Windows GUI-subsystem app) and can race ahead of file writes.
  #
  # AffinityMask (optional): a CPU affinity bitmask (over logical processors) to
  # pin the Unity process to. 0 = don't pin (default; tests/regen use all cores).
  # benchmark.ps1 passes a mask that keeps timing runs on ONE CCD, so single-
  # thread numbers don't wobble between the V-Cache and frequency dies on a
  # dual-CCD X3D part (results are unaffected -- this only stabilises timing).
  param(
    [Parameter(Mandatory)][string[]]$Arguments,
    [Parameter(Mandatory)][string]$LogFile,
    [long]$AffinityMask = 0
  )
  $unity = Get-UnityPath
  Clear-StaleLock
  $root = Get-ProjectRoot
  Write-Host "Unity:   $unity"
  Write-Host "Project: $root"
  New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null
  $all = @("-batchmode", "-projectPath", $root, "-logFile", $LogFile) + $Arguments
  # -WindowStyle Hidden keeps the launched Unity.exe from grabbing foreground focus,
  # which otherwise kicks an exclusive-fullscreen game (e.g. CS2) out to the desktop.
  $proc = Start-Process -FilePath $unity -ArgumentList $all -PassThru -WindowStyle Hidden
  if ($AffinityMask -ne 0) {
    try {
      $proc.ProcessorAffinity = [IntPtr][long]$AffinityMask
      Write-Host ("Pinned Unity (PID {0}) to CPU affinity 0x{1:X}" -f $proc.Id, $AffinityMask)
    } catch {
      Write-Host "WARN: could not set processor affinity (0x$($AffinityMask.ToString('X'))): $_"
    }
  }
  # Hang guard: bound the blocking wait so a stalled editor is killed and reported
  # as a failure instead of blocking forever (a genuine Unity hang, or a crashed
  # child that never releases). Override the ceiling via $env:UNITY_TIMEOUT_SEC.
  $timeoutSec = if ($env:UNITY_TIMEOUT_SEC) { [int]$env:UNITY_TIMEOUT_SEC } else { 900 }
  if (-not $proc.WaitForExit($timeoutSec * 1000)) {
    Write-Host "ERROR: Unity exceeded ${timeoutSec}s wall-clock -- killing (hang guard)."
    try { $proc.Kill() } catch {}
    Start-Sleep -Milliseconds 500
    return 124
  }
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
