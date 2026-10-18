$ErrorActionPreference = 'Stop'
$env:UNITY_MCP_TELEMETRY_ENABLED = 'false'
$uvx = Join-Path $env:USERPROFILE '.local\bin\uvx.exe'
if (-not (Test-Path -LiteralPath $uvx)) { throw 'uvx.exe not found.' }
$projectName = Split-Path (Split-Path -Parent $PSScriptRoot) -Leaf
& $uvx --from 'mcpforunityserver==9.6.7b20260427012228' mcp-for-unity --transport http --http-url 'http://127.0.0.1:8080' --project-scoped-tools --default-instance $projectName
