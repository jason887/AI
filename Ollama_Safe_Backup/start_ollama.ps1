$ErrorActionPreference = 'Stop'

param(
  [string]$Model = 'qwen3.5:9b',
  [string]$HostAddr = '127.0.0.1:11434'
)

$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
$env:OLLAMA_HOST = $HostAddr

Get-Process ollama -ErrorAction SilentlyContinue | ForEach-Object {
  try { Stop-Process -Id $_.Id -Force -ErrorAction Stop } catch {}
}

$p = Start-Process -FilePath 'ollama' -ArgumentList @('serve') -PassThru

$hostParts = $HostAddr.Split(':', 2)
$ip = $hostParts[0]
$port = [int]$hostParts[1]

$ready = $false
for ($i = 0; $i -lt 60; $i++) {
  $tcp = Test-NetConnection -ComputerName $ip -Port $port -WarningAction SilentlyContinue
  if ($tcp.TcpTestSucceeded) { $ready = $true; break }
  Start-Sleep -Seconds 1
}

if (-not $ready) {
  Write-Host "ollama serve 未在 ${HostAddr} 就绪（等待 60 秒超时）。pid=$($p.Id)"
  exit 1
}

Write-Host "ollama serve 已就绪：http://$HostAddr"
Write-Host "env: OLLAMA_NUM_PARALLEL=$env:OLLAMA_NUM_PARALLEL OLLAMA_MAX_LOADED_MODELS=$env:OLLAMA_MAX_LOADED_MODELS OLLAMA_CONTEXT_LENGTH=$env:OLLAMA_CONTEXT_LENGTH"

ollama list
ollama show $Model | Select-Object -First 80
ollama run $Model "你好"
