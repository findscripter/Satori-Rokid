param(
    [string]$Question = '我现在身处什么环境？周围有哪些东西？我的双手正在做什么？哪些判断还不能确定？',
    [string]$Session = 'Research/Data/real-task/clean_surface_0_dense_2fps/session.json',
    [int]$AtMs = 9000,
    [int]$Window = 8,
    [string]$Output,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$projectDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$evidenceDir = Join-Path $projectDir 'Evidence'
$tunnelInfoPath = Join-Path $evidenceDir 'mage-vl-tunnel.json'
$sshExecutable = (Get-Command ssh.exe -CommandType Application | Select-Object -First 1).Source
$listeners = @(Get-NetTCPConnection -LocalPort 18866 -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    if (-not (Test-Path -LiteralPath $tunnelInfoPath)) { throw 'Local port 18866 is occupied without this project tunnel record.' }
    $tunnelInfo = Get-Content -LiteralPath $tunnelInfoPath -Raw | ConvertFrom-Json
    $runningTunnel = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$tunnelInfo.pid)"
    if (-not $runningTunnel -or $runningTunnel.Name -ne 'ssh.exe' -or
        $runningTunnel.CommandLine -notlike '*127.0.0.1:18866:127.0.0.1:18866*' -or
        $runningTunnel.CommandLine -notmatch '\bgpu-host\b' -or
        @($listeners | Where-Object { $_.OwningProcess -ne [int]$tunnelInfo.pid }).Count -gt 0) {
        throw 'Local port 18866 does not belong to the expected gpu-host SSH tunnel.'
    }
} else {
    $backendPython = Join-Path $projectDir 'Research\Backend\.venv\Scripts\python.exe'
    $launcherPath = Join-Path $projectDir 'Tools\start_mage_tunnel.py'
    if (-not (Test-Path -LiteralPath $backendPython) -or -not (Test-Path -LiteralPath $launcherPath)) {
        throw 'Project Python or detached SSH tunnel launcher is missing.'
    }
    $launchOutput = & $backendPython $launcherPath --project-root $projectDir
    if ($LASTEXITCODE -ne 0) {
        throw 'Detached SSH startup failed; inspect Evidence/mage-vl-tunnel-attempts. Existing process records were preserved.'
    }
    $launchResult = ($launchOutput -join [Environment]::NewLine) | ConvertFrom-Json
    if ($launchResult.status -ne 'ready' -or -not $launchResult.detached_verified -or -not $launchResult.listener_verified) {
        throw 'SSH launcher did not verify a detached, owned loopback listener.'
    }
}

# SSH captures the private credential directly into process memory. Never print it.
$serviceKey = (& $sshExecutable -T -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 gpu-host 'cat /home/gpu-user/work/mage-vl-satori/.secrets/api.token' | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $serviceKey.Length -lt 16 -or $serviceKey -match '\s') {
    throw 'Could not load the private Mage-VL service credential over SSH.'
}
$previousKey = [Environment]::GetEnvironmentVariable('MAGE_VL_API_KEY','Process')
try {
    $env:MAGE_VL_API_KEY = $serviceKey
    if ($CheckOnly) {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:18866/health' -Headers @{Authorization="Bearer $serviceKey"} -TimeoutSec 20
        $health | ConvertTo-Json -Depth 8
    } else {
        if (-not $Output) { $Output = 'Research/Backend/runs/mage-vl-context-' + (Get-Date -Format 'yyyyMMdd-HHmmssfff') }
        Push-Location -LiteralPath $projectDir
        try {
            & (Join-Path $projectDir 'Research/Backend/.venv/Scripts/python.exe') -m satori_repro ask-context `
                --base-url 'http://127.0.0.1:18866/v1' --model 'Mage-VL' `
                --session $Session --at-ms $AtMs --window $Window --question $Question `
                --output $Output --api-key-env MAGE_VL_API_KEY --timeout 180
            if ($LASTEXITCODE -ne 0) { throw 'Mage-VL question failed; inspect the new run directory.' }
        } finally { Pop-Location }
    }
} finally {
    if ($null -eq $previousKey) { Remove-Item Env:MAGE_VL_API_KEY -ErrorAction SilentlyContinue }
    else { $env:MAGE_VL_API_KEY = $previousKey }
    $serviceKey = $null
}
