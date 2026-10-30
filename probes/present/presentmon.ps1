# Runs one probe case under PresentMon and summarises each SEGMENT the probe prints:
# presentation mode counts, and for pacing segments ("T <ms> ms <mode>") the spacing of
# displayed frames against T/4. PresentMon needs an elevated prompt.
#
#   .\presentmon.ps1 flip
#   .\presentmon.ps1 pacing
param(
    [string]$Case = "flip",
    [string]$PresentMon = "C:\Program Files\NVIDIA Corporation\FrameViewSDK\bin\PresentMon_x64.exe",
    [string]$Exe = "$PSScriptRoot\..\..\target\probes\release\present.exe"
)
$ErrorActionPreference = "Stop"
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "PresentMon needs an elevated prompt" }
$out = Join-Path $env:TEMP "present-probe-$Case"
New-Item -ItemType Directory -Force $out | Out-Null
$csv = Join-Path $out "presentmon.csv"
$log = Join-Path $out "probe.txt"
Remove-Item -ErrorAction SilentlyContinue $csv
$pm = Start-Process -PassThru -WindowStyle Hidden $PresentMon -ArgumentList @("--process_name", "present.exe", "--output_file", $csv, "--qpc_time", "--session_name", "orrprobe", "--stop_existing_session", "--no_console_stats")
Start-Sleep 2
& $Exe $Case | Tee-Object -FilePath $log
Start-Sleep 1
& $PresentMon --session_name orrprobe --terminate_existing_session | Out-Null
$pm.WaitForExit(5000) | Out-Null

$rows = Import-Csv $csv
$qcol = ($rows[0].PSObject.Properties.Name | Where-Object { $_ -match "QPC" } | Select-Object -First 1)
$dcol = ($rows[0].PSObject.Properties.Name | Where-Object { $_ -match "BetweenDisplayChange" } | Select-Object -First 1)
"columns: qpc=$qcol display=$dcol rows=$($rows.Count)"
foreach ($line in Get-Content $log | Where-Object { $_ -like "SEGMENT *" }) {
    if ($line -notmatch "^SEGMENT (.+) qpc (\d+) (\d+)$") { continue }
    $label, $a, $b = $Matches[1], [int64]$Matches[2], [int64]$Matches[3]
    $seg = $rows | Where-Object { $q = [int64]$_.$qcol; $q -ge $a -and $q -le $b }
    $modes = ($seg | Group-Object PresentMode | ForEach-Object { "$($_.Name) x$($_.Count)" }) -join ", "
    $text = "SEGMENT $label | $($seg.Count) presents | $modes"
    if ($label -match "^T ([\d.]+) ms" -and $dcol) {
        $ideal = [double]$Matches[1] / 4
        $d = @($seg | ForEach-Object { [double]$_.$dcol } | Where-Object { $_ -gt 0 } | Sort-Object)
        if ($d.Count -gt 0) {
            $within = @($d | Where-Object { [math]::Abs($_ - $ideal) -le 1.0 }).Count
            $text += " | display change median $($d[[int]($d.Count / 2)]) ms, within 1 ms of $ideal ms: $([math]::Round(100 * $within / $d.Count, 1))%"
        }
    }
    $text
}
