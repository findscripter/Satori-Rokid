"""Exercise the legacy shell parser used by a long-running Unity Editor."""

import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell is Windows-only")
def test_unity_fallback_shell_parses_chinese_entrypoint_without_execution():
    shell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not shell.is_file():
        pytest.skip("Windows PowerShell 5 is unavailable")
    root = Path(__file__).resolve().parents[3]
    # Parsing the real entrypoint triggers no SSH, credentials, HTTP or GPU work.
    code = (
        "[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);"
        "$tokens=$null;$parseErrors=$null;"
        "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
        "(Join-Path (Get-Location) 'Tools/Ask-MageVL.ps1'),[ref]$tokens,[ref]$parseErrors);"
        "@{errors=@($parseErrors|ForEach-Object {$_.Message});"
        "question=$ast.ParamBlock.Parameters[0].DefaultValue.Value}|ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [str(shell), "-NoProfile", "-NonInteractive", "-Command", code],
        cwd=root, capture_output=True, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    parsed = json.loads(result.stdout.decode("utf-8-sig"))
    assert parsed["errors"] == []
    assert parsed["question"] == "我现在身处什么环境？周围有哪些东西？我的双手正在做什么？哪些判断还不能确定？"
