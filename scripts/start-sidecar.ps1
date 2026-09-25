param(
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $Candidates = @(
        (Join-Path $PSScriptRoot "..\config.json"),
        (Join-Path $PSScriptRoot "..\..\data\roleplay-kernel\config.json")
    )
    $ConfigPath = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    throw "config.json was not found; pass -ConfigPath explicitly"
}
$ResolvedConfig = (Resolve-Path -LiteralPath $ConfigPath).Path
$LocalPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $LocalPython) { $LocalPython } else { "python" }

& $Python -m roleplay_kernel.sidecar --config $ResolvedConfig
exit $LASTEXITCODE
