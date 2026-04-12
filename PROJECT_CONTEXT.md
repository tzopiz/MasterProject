# Контекст проекта для разработки и агентов

Расширенная справка по репозиторию **MasterProject** (AI Doctor). Дополняет [README.md](README.md) и [AGENTS.md](AGENTS.md).

## Назначение продукта

Система поддерживает сценарий **анализа височно-нижнечелюстных суставов (ВНЧС / TMJ)** по объёмным **КЛКТ (CBCT)** данным в формате **DICOM**: загрузка серии, вызов ML-сервиса, возврат координат/областей для визуализации в iOS-клиенте.

## Глоссарий

| Термин | Пояснение |
|--------|-----------|
| ВНЧС / TMJ | Височно-нижнечелюстной сустав |
| КЛКТ / CBCT | Конусно-лучевая КТ |
| DICOM | Стандарт медицинских изображений; в проекте — серии `.dcm` |
| Детектор | Модель локализации суставов в 3D-объёме (координаты, bounding box) |
| Сегментация | Отдельный этап в ML-тулчейне (U-Net и др.), см. MLService README |
| ROI | Region of interest; разметка и кропы вокруг сустава |

## Логическая архитектура

```text
iOS (SwiftUI)  --HTTP-->  Backend (Vapor + SQLite)  --HTTP-->  ML Service (FastAPI + PyTorch)
```

Типичный поток:

1. Клиент отправляет на Backend **серию DICOM** (multipart).
2. Backend сохраняет файлы, создаёт задачу, в фоне вызывает ML Service.
3. ML Service строит объём, прогоняет модель, возвращает координаты и метаданные (в т.ч. размер объёма).
4. Backend сохраняет результат в БД; клиент опрашивает или получает результат по `taskId`.

## Карта каталогов (верхний уровень)

```text
MasterProject/
├── Backend/
├── MLService/
├── iOSApp/MasterDoctor/
├── README.md
├── PROJECT_CONTEXT.md
├── TMJ_DETECTION_SETUP.md
└── AGENTS.md
```

Уточнения по ML: структура `MLService/data/`, чекпойнты в `experiments/` — см. [MLService/models/README.md](MLService/models/README.md), [MLService/experiments/README.md](MLService/experiments/README.md), [MLService/google_colab/POSITION_CLASSIFIER_EXPERIMENTS.md](MLService/google_colab/POSITION_CLASSIFIER_EXPERIMENTS.md).

## HTTP API (актуальность)

Всегда сверяйтесь с кодом. Ниже — ориентир по **Backend** (`Backend/Sources/App/Controllers/`).

| Метод | Путь | Назначение |
|-------|------|------------|
| GET | `/health` | Проверка живости Backend |
| POST | `/api/analysis` | Загрузка **multipart**: поле файлов согласно `AnalysisController` (`SeriesUploadRequest`, поле `files`) |
| GET | `/api/analysis/{taskId}` | Статус задачи и результаты (см. `AnalysisResponse` в `AnalysisController.swift`) |

Отдельного `GET /api/analysis/{taskId}/status` может не быть: статус приходит в составе `GET .../analysis/{taskId}`.

**ML Service** (`MLService/app.py`): типичные пути `/health`, `/models/status`, `/process`.

## Переменные окружения

| Сервис | Переменная | Смысл |
|--------|------------|--------|
| Backend | `ML_SERVICE_URL` | Базовый URL ML Service (часто `http://localhost:8001`) |
| ML Service | `MODEL_PATH` | Путь к весам детектора (по умолчанию см. `app.py`, ожидается `models/tmj_detector_best.pth`) |

## Локальные данные: публичная CBCT-когорта

Скрипты в `MLService/tools/`, каталоги **`MLService/data/cbct_public_zips/`** и **`MLService/data/cbct_public_extracted/`** в `.gitignore`; метки — **`MLService/data/tmj_position_labels.json`**. Подробные методички при необходимости ведите отдельно от репозитория.

## Соглашения репозитория

- Секреты и лишние артефакты не коммитятся (`.gitignore`, `MLService/.gitignore`). Веса в **`MLService/models/`** можно держать в git; тяжёлые прогоны — в `MLService/experiments/` (по правилам игнора); локальный **`models/`** в корне репозитория — вне git.
- Каталог `.cursor/` не версионируется.
