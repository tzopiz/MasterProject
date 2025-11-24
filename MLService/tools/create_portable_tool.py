import os
import shutil
from pathlib import Path

def create_portable_package():
    # 1. Setup paths
    root_dir = Path(__file__).parent.parent
    source_tool = root_dir / 'tools' / 'roi_annotation_tool.py'
    
    dist_dir = root_dir / 'TMJ_Annotator_Portable'
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()
    
    print(f"📦 Creating portable package in: {dist_dir}")

    # 2. Copy Python Tool
    shutil.copy2(source_tool, dist_dir / 'roi_annotation_tool.py')
    print("✅ Copied roi_annotation_tool.py")

    # 3. Create requirements.txt (Minimal)
    reqs = """numpy
matplotlib
pydicom
"""
    with open(dist_dir / 'requirements.txt', 'w') as f:
        f.write(reqs)
    print("✅ Created requirements.txt")

    # 4. Create Usage Guide
    readme = """# TMJ Annotation Tool

## Поддерживаемые платформы:
- Windows 10/11
- macOS (Intel/Silicon)
- Linux

## Как запустить:

### 🖥️ Windows:
1. Дважды кликни по файлу `run_windows.bat`.
2. Если появится окно "Windows защитила ваш компьютер", нажми "Подробнее" -> "Выполнить в любом случае".
3. Следуй инструкциям в черном окне консоли.

### 🍎 macOS:
1. Дважды кликни по файлу `run_mac.command`.
2. Если macOS заблокирует запуск, зайди в Системные настройки -> Конфиденциальность и безопасность -> Разрешить выполнение.

### 🐧 Linux:
1. Запусти файл `run_linux.sh` из терминала: `bash run_linux.sh`.

## Инструкция:
1. Скрипт попросит перетащить папку с исследованиями в окно.
2. Затем попросит указать папку для сохранения результатов.
3. Откроется окно с КТ.
   - **Колесико мыши**: Листать срезы.
   - **ЛКМ (Левый клик)**: Поставить точку центра сустава.
   - **ПКМ (Правый клик)**: Удалить точку.
   - **Q**: Сохранить и перейти к следующему.
   - **Esc**: Пропустить.

## Важно:
- На компьютере должен быть установлен **Python** (версии 3.8 или новее).
- Программа сама создаст виртуальное окружение и установит библиотеки при первом запуске.
"""
    with open(dist_dir / 'README.txt', 'w') as f:
        f.write(readme)
    print("✅ Created README.txt")

    # 5. Create macOS/Linux Runner
    runner_sh = """#!/bin/bash
cd "$(dirname "$0")"

echo "==============================================="
echo "       TMJ ANNOTATION TOOL SETUP"
echo "==============================================="

# 1. Setup Python Environment
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    # Try python3 first, then python
    if command -v python3 &>/dev/null; then
        python3 -m venv venv
    else
        python -m venv venv
    fi
fi

echo "🔌 Activating environment..."
source venv/bin/activate

echo "📦 Checking dependencies..."
pip install -r requirements.txt -q

echo ""
echo "==============================================="
echo "       READY TO ANNOTATE"
echo "==============================================="
echo ""

# 2. Get Input Directory
while true; do
    echo "📂 Drag and drop the folder containing ALL studies here, then press Enter:"
    read -r INPUT_DIR
    # Remove quotes if present (happens with drag & drop)
    INPUT_DIR="${INPUT_DIR%\\"}"
    INPUT_DIR="${INPUT_DIR#\\"}"
    # Trim whitespace
    INPUT_DIR="$(echo "$INPUT_DIR" | xargs)"
    
    if [ -d "$INPUT_DIR" ]; then
        break
    else
        echo "❌ Invalid directory. Please try again."
    fi
done

# 3. Get Output Directory
echo ""
echo "📂 Drag and drop folder for saving JSONs (or press Enter to save inside input folder):"
read -r OUTPUT_DIR
# Remove quotes
OUTPUT_DIR="${OUTPUT_DIR%\\"}"
OUTPUT_DIR="${OUTPUT_DIR#\\"}"
OUTPUT_DIR="$(echo "$OUTPUT_DIR" | xargs)"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$INPUT_DIR/annotations"
    mkdir -p "$OUTPUT_DIR"
    echo "Using default output: $OUTPUT_DIR"
fi

# 4. Iterate through folders
echo ""
echo "🚀 Starting annotation session..."
echo "Press 'Q' in the window to save and go to next."
echo "Press 'Esc' to skip."
echo ""

count=0
for study_path in "$INPUT_DIR"/*; do
    if [ -d "$study_path" ]; then
        study_name=$(basename "$study_path")
        
        if [ -f "$OUTPUT_DIR/${study_name}_rois.json" ]; then
            echo "⏩ Skipping $study_name (already done)"
            continue
        fi
        
        echo "----------------------------------------"
        echo "Processing: $study_name"
        
        # Check for DICOMs (rudimentary check)
        # Just check if directory is not empty of files
        if [ -z "$(ls -A "$study_path")" ]; then
             echo "⚠️ Empty folder, skipping..."
             continue
        fi
        
        python roi_annotation_tool.py \
            --scan_id "$study_name" \
            --dicom_dir "$study_path" \
            --output_dir "$OUTPUT_DIR"
            
        ((count++))
    fi
done

echo ""
echo "🎉 Done! Processed $count studies."
echo "Results saved in: $OUTPUT_DIR"
echo "Press any key to exit..."
read -n 1
"""
    run_mac = dist_dir / 'run_mac.command'
    with open(run_mac, 'w') as f:
        f.write(runner_sh)
    os.chmod(run_mac, 0o755)
    print("✅ Created run_mac.command")
    
    run_linux = dist_dir / 'run_linux.sh'
    with open(run_linux, 'w') as f:
        f.write(runner_sh)
    os.chmod(run_linux, 0o755)
    print("✅ Created run_linux.sh")

    # 6. Create Windows Batch Script
    runner_bat = r"""@echo off
setlocal EnableDelayedExpansion

echo ===============================================
echo        TMJ ANNOTATION TOOL SETUP
echo ===============================================

REM 1. Setup Python Environment
if not exist "venv" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Error: Python not found or failed to create venv.
        echo Please install Python from https://www.python.org/downloads/
        pause
        exit /b
    )
)

echo 🔌 Activating environment...
call venv\Scripts\activate.bat

echo 📦 Checking dependencies...
pip install -r requirements.txt -q

echo.
echo ===============================================
echo        READY TO ANNOTATE
echo ===============================================
echo.

REM 2. Get Input Directory
:ask_input
echo 📂 Drag and drop the folder containing ALL studies here, then press Enter:
set /p INPUT_DIR=
REM Remove quotes
set INPUT_DIR=%INPUT_DIR:"=%

if not exist "%INPUT_DIR%" (
    echo ❌ Invalid directory. Please try again.
    goto ask_input
)

REM 3. Get Output Directory
echo.
echo 📂 Drag and drop folder for saving JSONs (or press Enter to save inside input folder):
set /p OUTPUT_DIR=
if "%OUTPUT_DIR%"=="" (
    set "OUTPUT_DIR=%INPUT_DIR%\annotations"
) else (
    set OUTPUT_DIR=%OUTPUT_DIR:"=%
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
echo Using output: "%OUTPUT_DIR%"

REM 4. Iterate through folders
echo.
echo 🚀 Starting annotation session...
echo Press 'Q' in the window to save and go to next.
echo Press 'Esc' to skip.
echo.

set count=0
for /d %%D in ("%INPUT_DIR%\*") do (
    set "study_path=%%D"
    set "study_name=%%~nxD"
    
    if exist "%OUTPUT_DIR%\!study_name!_rois.json" (
        echo ⏩ Skipping !study_name! (already done)
    ) else (
        echo ----------------------------------------
        echo Processing: !study_name!
        
        python roi_annotation_tool.py --scan_id "!study_name!" --dicom_dir "!study_path!" --output_dir "%OUTPUT_DIR%"
        
        set /a count+=1
    )
)

echo.
echo 🎉 Done! Processed !count! studies.
echo Results saved in: "%OUTPUT_DIR%"
echo Press any key to exit...
pause >nul
"""
    run_bat = dist_dir / 'run_windows.bat'
    with open(run_bat, 'w') as f:
        f.write(runner_bat)
    print("✅ Created run_windows.bat")
    
    print(f"\n🚀 SUCCESS! Folder '{dist_dir}' is ready.")
    print("It supports macOS, Linux, and Windows.")

if __name__ == "__main__":
    create_portable_package()
