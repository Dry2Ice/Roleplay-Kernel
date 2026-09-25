param(
    [Parameter(Mandatory = $true)]
    [string]$SillyTavernPath,
    [string]$UpstreamBaseUrl = "https://api.openai.com/v1",
    [string]$UpstreamModel = "gpt-4o-mini",
    [string]$ApiKeyEnv = "OPENAI_API_KEY",
    [int]$Port = 8787,
    [switch]$AllowInsecureHttp,
    [switch]$SkipServerPluginConfig,
    [switch]$StartStandalone,
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$SillyTavernRoot = (Resolve-Path -LiteralPath $SillyTavernPath).Path
$ExtensionParent = Join-Path $SillyTavernRoot "public\scripts\extensions\third-party"
$ExtensionTarget = Join-Path $ExtensionParent "RoleplayKernel"
$RuntimeRoot = Join-Path $SillyTavernRoot "data\roleplay-kernel"
$VirtualEnvironment = Join-Path $RuntimeRoot "venv"
$Python = Join-Path $VirtualEnvironment "Scripts\python.exe"
$ConfigPath = Join-Path $RuntimeRoot "config.json"
$PluginsRoot = Join-Path $SillyTavernRoot "plugins"
$ServerPluginTarget = Join-Path $PluginsRoot "roleplay-kernel"
$ServerPluginSource = Join-Path $RepoRoot "server-plugin"

if (-not (Test-Path -LiteralPath $ExtensionParent)) {
    throw "SillyTavern extension directory not found: $ExtensionParent"
}
if (-not (Test-Path -LiteralPath (Join-Path $SillyTavernRoot "Start.bat"))) {
    throw "The supplied path does not look like a SillyTavern installation"
}
if (-not (Test-Path -LiteralPath (Join-Path $ServerPluginSource "index.js"))) {
    throw "server-plugin/index.js was not found; run the installer from the source checkout"
}

$RepoIsUiInstall = $RepoRoot.StartsWith(
    $ExtensionParent,
    [System.StringComparison]::OrdinalIgnoreCase
)
if (-not $RepoIsUiInstall) {
    New-Item -ItemType Directory -Path $ExtensionTarget -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $ExtensionTarget "i18n") -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $RepoRoot "manifest.json") -Destination $ExtensionTarget -Force
    Copy-Item -LiteralPath (Join-Path $RepoRoot "index.js") -Destination $ExtensionTarget -Force
    Copy-Item -LiteralPath (Join-Path $RepoRoot "style.css") -Destination $ExtensionTarget -Force
    Copy-Item -LiteralPath (Join-Path $RepoRoot "settings.html") -Destination $ExtensionTarget -Force
    Copy-Item -LiteralPath (Join-Path $RepoRoot "i18n\ru-ru.json") -Destination (Join-Path $ExtensionTarget "i18n") -Force
}
New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ServerPluginTarget -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $ServerPluginSource "index.js") -Destination $ServerPluginTarget -Force

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $VirtualEnvironment
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create the Python virtual environment"
}
& $Python -m pip install --disable-pip-version-check $RepoRoot
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Roleplay Kernel"
}

$PythonW = Join-Path $VirtualEnvironment "Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $PythonW)) {
    $PythonW = $Python
}
$RuntimeSettings = [ordered]@{
    python = $PythonW
    module = "roleplay_kernel.sidecar"
    configPath = $ConfigPath
    cwd = $RuntimeRoot
}
$RuntimeSettingsJson = $RuntimeSettings | ConvertTo-Json -Depth 3
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $ServerPluginTarget "runtime.json"), $RuntimeSettingsJson, $Utf8NoBom)

if (-not $SkipServerPluginConfig) {
    $StConfigPath = Join-Path $SillyTavernRoot "config.yaml"
    if (Test-Path -LiteralPath $StConfigPath) {
        $StConfigText = Get-Content -LiteralPath $StConfigPath -Raw
        if ($StConfigText -match '(?m)^enableServerPlugins\s*:') {
            $StConfigText = [regex]::Replace(
                $StConfigText,
                '(?m)^enableServerPlugins\s*:\s*.*$',
                'enableServerPlugins: true'
            )
        } else {
            $StConfigText = $StConfigText.TrimEnd() + "`nenableServerPlugins: true`n"
        }
        [System.IO.File]::WriteAllText($StConfigPath, $StConfigText, $Utf8NoBom)
    }
}

$IntegrationKey = ""
$NeedsConfig = $true
$EffectiveUpstreamBaseUrl = $UpstreamBaseUrl
$EffectiveUpstreamModel = $UpstreamModel
$EffectiveApiKeyEnv = $ApiKeyEnv
$EffectiveAllowInsecureHttp = [bool]$AllowInsecureHttp
if (Test-Path -LiteralPath $ConfigPath) {
    try {
        $ExistingConfig = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
        $ExistingKey = [string]$ExistingConfig.integration_key
        if ($ExistingKey.Length -ge 32) {
            $IntegrationKey = $ExistingKey
            $NeedsConfig = $false
            if ($null -ne $ExistingConfig.upstream_base_url) {
                $EffectiveUpstreamBaseUrl = [string]$ExistingConfig.upstream_base_url
            }
            if ($null -ne $ExistingConfig.upstream_model) {
                $EffectiveUpstreamModel = [string]$ExistingConfig.upstream_model
            }
            if ($null -ne $ExistingConfig.upstream_api_key_env) {
                $EffectiveApiKeyEnv = [string]$ExistingConfig.upstream_api_key_env
            }
            if ($null -ne $ExistingConfig.allow_insecure_http) {
                $EffectiveAllowInsecureHttp = [bool]$ExistingConfig.allow_insecure_http
            }
        }
    } catch {
        throw "Existing config is not valid JSON: $ConfigPath"
    }
}
if ($NeedsConfig) {
    $KeyBytes = New-Object byte[] 32
    $Random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $Random.GetBytes($KeyBytes)
    $Random.Dispose()
    $IntegrationKey = [Convert]::ToBase64String($KeyBytes)
    $Config = [ordered]@{
        host = "127.0.0.1"
        port = $Port
        upstream_base_url = $UpstreamBaseUrl.TrimEnd("/")
        upstream_model = $UpstreamModel
        upstream_api_key_env = $ApiKeyEnv
        upstream_token_parameter = "max_tokens"
        integration_key = $IntegrationKey
        state_dir = (Join-Path $RuntimeRoot "state")
        mode = "balanced"
        context_window = 32768
        token_budget = 18000
        max_output_tokens = 1200
        max_internal_tokens = 1200
        max_repairs = 1
        max_context_chars = 16000
        allow_insecure_http = [bool]$AllowInsecureHttp
    }
    $ConfigJson = $Config | ConvertTo-Json -Depth 4
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($ConfigPath, $ConfigJson, $Utf8NoBom)
}

if ($env:OS -eq "Windows_NT") {
    $CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    & icacls $ConfigPath /inheritance:r /grant:r "$($CurrentUser):(F)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to restrict config permissions"
    }
}

$LauncherPath = Join-Path $RuntimeRoot "Start-RoleplayKernel.ps1"
$LauncherContent = @'
$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
& $Python -m roleplay_kernel.sidecar --config (Join-Path $PSScriptRoot "config.json")
exit $LASTEXITCODE
'@
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($LauncherPath, $LauncherContent, $Utf8NoBom)

$PreviousValidationPath = $env:RPK_CONFIG_VALIDATION_PATH
$env:RPK_CONFIG_VALIDATION_PATH = $ConfigPath
try {
    & $Python -c "import os; from pathlib import Path; from roleplay_kernel.sidecar import SidecarConfig; SidecarConfig.load(Path(os.environ['RPK_CONFIG_VALIDATION_PATH']))"
    if ($LASTEXITCODE -ne 0) {
        throw "Roleplay Kernel config validation failed"
    }
} finally {
    $env:RPK_CONFIG_VALIDATION_PATH = $PreviousValidationPath
}

Write-Host "Roleplay Kernel installed"
Write-Host "Integration key: $IntegrationKey"
Write-Host "Enter this key in the extension settings."
Write-Host "Config: $ConfigPath"
Write-Host "Launcher: $LauncherPath"

if ($StartStandalone -and -not $NoStart) {
    $EnvironmentValue = Get-Item -LiteralPath "Env:$EffectiveApiKeyEnv" -ErrorAction SilentlyContinue
    $MissingApiKey = $null -eq $EnvironmentValue -or [string]::IsNullOrWhiteSpace($EnvironmentValue.Value)
    if ($EffectiveUpstreamBaseUrl.StartsWith("https://") -and $MissingApiKey) {
        Write-Warning "Environment variable $EffectiveApiKeyEnv is not set; direct upstream fallback will be unavailable"
    }
    & $Python -m roleplay_kernel.sidecar --config $ConfigPath
} else {
    Write-Host "Runtime is managed by the ST server plugin; use the Launch button in the extension."
}
