# TMJ Detection System - Setup Guide

## Overview

Система для автоматической детекции височно-нижнечелюстных суставов (TMJ) на 3D CBCT снимках.

## Architecture

```
iOS App (Swift) → Backend (Vapor) → ML Service (FastAPI) → TMJ Detector (PyTorch)
```

## Components

### 1. ML Service (Python FastAPI)
Расположение: `/MLService`

**Модель детектора:**
- Архитектура: TMJDetectorLarge (3D CNN, 14.4M параметров)
- Вход: 3D CBCT volume (downsampled to 96×128×128)
- Выход: координаты левого и правого TMJ [z, y, x]
- Лучшая модель: `experiments/detector_20251124_175805/best_model.pth`

**Запуск:**
```bash
cd MLService
source venv/bin/activate
python app.py
# или
./start.sh
```

**Endpoints:**
- `GET /health` - проверка статуса
- `GET /models/status` - статус модели
- `POST /process` - обработка DICOM серии

### 2. Backend (Swift Vapor)
Расположение: `/Backend`

**Endpoints:**
- `POST /api/analysis` - загрузка DICOM файлов
- `GET /api/analysis/:taskId/status` - статус обработки
- `GET /api/analysis/:taskId` - результаты детекции

**Запуск:**
```bash
cd Backend
swift run
# или
./start.sh
```

### 3. iOS App (SwiftUI)
Расположение: `/iOSApp/MasterDoctor`

**Новые возможности:**
- Загрузка папки с DICOM файлами
- Визуализация координат TMJ
- Отображение bounding box
- Информация о размере volume

**Запуск:**
- Открыть `MasterDoctor.xcodeproj` в Xcode
- Выбрать схему и запустить

## Обучение модели детектора

### Текущее обучение
Модель сейчас обучается в background:
```bash
# Проверить прогресс
tail -f ~/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/10.txt | grep "Epoch"
```

### Параметры обучения
- Датасет: 37 аннотированных снимков (31 train / 6 val)
- Batch size: 4
- Learning rate: 5e-5
- Epochs: 150 (с early stopping)
- Device: Apple Silicon MPS

### Запуск нового обучения
```bash
cd MLService
source venv/bin/activate

python train_detector.py \
  --annotations data/roi_annotations \
  --dataset data/dataset \
  --model_type large \
  --epochs 150 \
  --batch_size 4 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --downsample_factor 6 \
  --patience 15 \
  --early_stopping 40 \
  --split_ratio 0.85
```

## API Flow

### 1. Upload DICOM Series
```
POST /api/analysis
Content-Type: multipart/form-data

files: [file1.dcm, file2.dcm, ...]
```

Response:
```json
{
  "taskId": "uuid"
}
```

### 2. Check Status
```
GET /api/analysis/{taskId}/status
```

Response:
```json
{
  "taskId": "uuid",
  "status": "processing|completed|failed",
  "errorMessage": null
}
```

### 3. Get Results
```
GET /api/analysis/{taskId}
```

Response:
```json
{
  "taskId": "uuid",
  "status": "completed",
  "tmjLeft": "{\"center\": [z, y, x], \"bbox\": [z1, y1, x1, z2, y2, x2]}",
  "tmjRight": "{\"center\": [z, y, x], \"bbox\": [z1, y1, x1, z2, y2, x2]}",
  "volumeShape": [576, 768, 768],
  "createdAt": "2025-11-24T...",
  "updatedAt": "2025-11-24T..."
}
```

## Data Format

### BoundingBox
```json
{
  "center": [z, y, x],  // float coordinates in voxels
  "bbox": [z1, y1, x1, z2, y2, x2]  // integer bounding box
}
```

### Volume Shape
`[depth, height, width]` - размеры 3D volume в вокселях

## Database

SQLite база в `/Backend/db.sqlite`

Таблицы:
- `analysis_tasks` - задачи обработки
- `analysis_results` - результаты детекции

## Configuration

### ML Service
- URL: `http://localhost:8001`
- Environment: `ML_SERVICE_URL`

### Backend
- URL: `http://localhost:8080`
- Database: SQLite (development)

### iOS App
- Backend URL: `http://localhost:8080/api`
- Можно изменить в `AnalysisEndpoint.swift`

## Troubleshooting

### ML Service не загружает модель
```bash
# Проверить наличие модели
ls -lh experiments/detector_*/best_model.pth

# Указать путь явно
export MODEL_PATH="experiments/detector_20251124_175805/best_model.pth"
python app.py
```

### Backend не подключается к ML Service
```bash
# Проверить что ML Service запущен
curl http://localhost:8001/health

# Указать правильный URL
export ML_SERVICE_URL="http://localhost:8001"
```

### iOS App не подключается к Backend
- Проверить URL в `AnalysisEndpoint.swift`
- Убедиться что Backend запущен
- Для симулятора: использовать `http://localhost:8080`
- Для реального устройства: использовать IP адрес Mac

## Monitoring Training

### Проверка метрик
```bash
cd MLService
./venv/bin/python check_progress.py
```

### Просмотр логов обучения

Смотрите вывод процесса в том терминале, где запущено обучение, или файлы логов в каталоге эксперимента (если настроена запись в файл). Пример из файла терминала Cursor:

```bash
tail -50 ~/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/10.txt | grep -E "Epoch|MAE"
```

### Визуализация результатов детектора
```bash
cd MLService
./venv/bin/python tools/visualize_detector.py \
  --model experiments/detector_20251124_175805/best_model.pth \
  --dataset data/dataset \
  --annotations data/roi_annotations \
  --output visualizations/
```

## Next Steps

1. ✅ Backend готов для работы с множественными DICOM файлами
2. ✅ ML Service интегрирован с детектором TMJ
3. ✅ iOS клиент поддерживает загрузку папок и визуализацию
4. 🔄 Модель обучается на расширенном датасете
5. ⏭️ Добавить 3D визуализацию volume с наложением координат TMJ
6. ⏭️ Экспорт результатов в формат для дальнейшего анализа

## Model Performance

Текущие метрики (Epoch 3):
- Validation MAE: **97.34 px**
- Train MAE: 82.50 px

Цель: MAE < 50 px (sub-voxel accuracy)

---

Created: 2025-11-24
Last Updated: 2025-11-24

