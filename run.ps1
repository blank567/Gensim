param(
    [ValidateSet('all', 'prepare', 'explore', 'classify', 'source', 'verify', 'report', 'predict')]
    [string]$Command = 'all',
    [string]$Text = '',
    [ValidateSet('2', '4', '6')][string]$Classes = '4',
    [string]$Method = 'fasttext_tfidf'
)
$ErrorActionPreference = 'Stop'
$env:TEMP = 'E:\codex\tmp'
$env:TMP = 'E:\codex\tmp'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONHASHSEED = '42'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache\pip'
$pythonForGensim = 'E:\conda_envs\llm\python.exe'
if (-not (Test-Path -LiteralPath $pythonForGensim)) { throw "找不到指定的 llm 环境：$pythonForGensim" }
Push-Location $PSScriptRoot
try {
    $labArguments = @('-X', 'utf8', 'main.py', $Command, '--classes', $Classes, '--method', $Method)
    if ($Text) { $labArguments += @('--text', $Text) }
    & $pythonForGensim @labArguments
    if ($LASTEXITCODE -ne 0) { throw "实验失败，退出码：$LASTEXITCODE" }
} finally {
    Pop-Location
}
