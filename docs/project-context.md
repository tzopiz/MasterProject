# Контекст проекта для разработки и агентов

Расширенная справка по репозиторию **MasterProject** (AI Doctor). Дополняет [README.md](../README.md) и [AGENTS.md](../AGENTS.md).

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
├── Backend/           # Vapor, миграции Fluent, HTTP к MLService
├── MLService/         # FastAPI app.py, services/, models/, tools/, experiments/
├── iOSApp/            # Xcode: MasterDoctor и модули
├── docs/              # Хаб документации (README.md); подкаталог presentation/ — слайды и конференция
├── scripts/
│   └── presentation/  # create_* / export_* для DOCX и PDF/PPTX
├── examples/          # Примеры / вспомогательные данные
├── Nir/               # НИР, отчёты
├── README.md          # Главный обзор
├── QUICKSTART.md      # Запуск «с нуля» для демо детекции
├── TMJ_DETECTION_SETUP.md
├── AGENTS.md          # Короткая памятка для ИИ-агентов
```

Уточнения по ML: структура `MLService/data/`, чекпойнты в `experiments/`, игнорируемые тяжёлые артефакты — в `MLService/.gitignore` и [MLService/models/README.md](../MLService/models/README.md).

## HTTP API (актуальность)

Всегда сверяйтесь с кодом. Ниже — ориентир по **Backend** (`Backend/Sources/App/Controllers/`).

| Метод | Путь | Назначение |
|-------|------|------------|
| GET | `/health` | Проверка живости Backend |
| POST | `/api/analysis` | Загрузка **multipart**: тело с полем файлов, ожидаемым `AnalysisController` (тип `SeriesUploadRequest`, поле `files`) |
| GET | `/api/analysis/{taskId}` | Статус задачи и результаты (поля ответа — см. `AnalysisResponse` в `AnalysisController.swift`) |

Отдельного маршрута `GET /api/analysis/{taskId}/status` в текущем `AnalysisController` может не быть: статус приходит в составе `GET .../analysis/{taskId}`. Если iOS или другой клиент обращается к `/status`, проверьте соответствие клиента и сервера.

**ML Service** (ориентир — `MLService/app.py`): типичные пути `/health`, `/models/status`, `/process`; уточняйте декораторы роутов в приложении.

## Переменные окружения

| Сервис | Переменная | Смысл |
|--------|------------|--------|
| Backend | `ML_SERVICE_URL` | Базовый URL ML Service (часто `http://localhost:8001`) |
| ML Service | `MODEL_PATH` | Путь к весам детектора/модели |

## Документы с историческим уклоном

Некоторые файлы отражали более ранний API (например, JSON-загрузка одного файла). При противоречии с `AnalysisController` и `AnalysisEndpoint.swift` **приоритет у кода**. Имеет смысл постепенно выравнивать [Backend/README.md](../Backend/README.md) и корневой README.

## Локальные данные: публичная CBCT-когорта

Для задачи классификации положения головок ВНЧС (коды из клинического DOCX) используются скрипты в `MLService/tools/` и каталоги **`MLService/data/cbct_public_zips/`** (архивы) и **`MLService/data/cbct_public_extracted/`** (распаковка + очистка под DICOM). Они **в `.gitignore`**; метки в репозитории — **`MLService/data/tmj_position_labels.json`** (генерируется из DOCX). Подробно: [cbct-public-cohort-dataset.md](cbct-public-cohort-dataset.md).

## Соглашения репозитория

- Секреты, большие веса и локальные артефакты не коммитятся (см. корневой `.gitignore`, `MLService/.gitignore`).
- Личные PDF и сгенерированные офисные файлы (в т.ч. в `docs/presentation/`) перечислены в `.gitignore`; источники — Markdown/HTML и скрипты в `scripts/presentation/`.
