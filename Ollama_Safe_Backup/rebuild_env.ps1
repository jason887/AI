# 1. 进入目标目录
cd "F:\Ollama_Safe_Backup"

# 2. 删除可能存在的旧环境残余
if (Test-Path "venv") {
    Write-Host "正在清理旧环境..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

# 3. 创建全新的虚拟环境
Write-Host "正在创建全新虚拟环境..." -ForegroundColor Cyan
python -m venv venv

# 4. 激活新环境
Write-Host "正在激活环境..." -ForegroundColor Cyan
.\venv\Scripts\activate

# 5. 修复并升级 pip
Write-Host "正在修复 pip..." -ForegroundColor Cyan
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# 6. 一键安装所有采集必备依赖
Write-Host "正在安装 OCR 与自动化依赖（这可能需要几分钟）..." -ForegroundColor Cyan
pip install pillow numpy paddlepaddle paddleocr easyocr ddddocr -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "--- 环境重建完毕！---" -ForegroundColor Green
Write-Host "现在你可以运行: python tools\final_worker.py" -ForegroundColor White