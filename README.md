# AI Doctor - Анализ ВНЧС

Магистерский проект: iOS приложение с ИИ для анализа височно-нижнечелюстного сустава (ВНЧС) по снимкам КЛКТ.

**Документация:** [docs/README.md](docs/README.md) (оглавление) · для ИИ-агентов: [AGENTS.md](AGENTS.md) · контекст: [docs/project-context.md](docs/project-context.md)

## Архитектура

Проект состоит из трех компонентов:

1. **Backend** (Swift Vapor) - API для управления задачами и данными
2. **MLService** (Python FastAPI) - ML инференс и обработка DICOM
3. **iOS App** (Swift/SwiftUI) — клиент (модульная структура, см. [iOSApp/ModularAppArchitecture.md](iOSApp/ModularAppArchitecture.md))

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   iOS App   │ ◄─────► │    Backend   │ ◄─────► │  ML Service │
│  (Swift)    │   API   │   (Vapor)    │   API   │  (FastAPI)  │
└─────────────┘         └──────────────┘         └─────────────┘
                              │
                              ▼
                        ┌──────────┐
                        │ SQLite   │
                        │    DB    │
                        └──────────┘
```

## Технологический стек

### Backend (Vapor)
- Swift 5.9+
- Vapor 4.x - веб-фреймворк
- Fluent + SQLite - ORM и база данных
- AsyncHTTPClient - HTTP клиент

### ML Service
- Python 3.9+
- FastAPI - API фреймворк
- PyTorch - ML инференс
- pydicom - обработка DICOM
- scikit-image, OpenCV - обработка изображений

### iOS App
- Swift 5.9+
- SwiftUI
- Сетевой слой (например `AnalysisEndpoint`) — базовый URL API настраивается в коде клиента

## Быстрый старт

### Предварительные требования

- Swift 5.9+ и Xcode (для Backend)
- Python 3.9+ (для ML Service)
- macOS 13+ или Linux

### 1. Запуск Backend

```bash
cd Backend
swift build
swift run App
```

Backend запустится на `http://localhost:8080`

### 2. Запуск ML Service

```bash
cd MLService
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

ML Service запустится на `http://localhost:8001`

### 3. Проверка работоспособности

Health check Backend:
```bash
curl http://localhost:8080/health
```

Health check ML Service:
```bash
curl http://localhost:8001/health
```

## API

Актуальные контракты смотрите в коде (`Backend/Sources/App/Controllers/`, `MLService/app.py`). Кратко:

### Backend

- `GET /health` — health check
- `POST /api/analysis` — загрузка **серии** DICOM (`multipart/form-data`, поле с файлами согласно `SeriesUploadRequest` в `AnalysisController.swift`)
- `GET /api/analysis/{taskId}` — статус задачи и результаты (в одном ответе)

### ML Service

- `GET /health` — health check
- `GET /models/status` — статус модели
- `POST /process` — обработка (серия/объём; детали в FastAPI-приложении)

## Workflow

1. **Загрузка DICOM**: клиент (iOS) отправляет серию `.dcm` на Backend (`POST /api/analysis`).
2. **Backend**: сохраняет файлы в `uploads/`, создаёт задачу в SQLite, в фоне вызывает ML Service.
3. **ML Service**: собирает 3D-объём, выполняет **детекцию** TMJ (координаты, при необходимости bbox), может включать дополнительную обработку по текущему пайплайну.
4. **Backend**: сохраняет JSON-результаты и метаданные (в т.ч. `volumeShape`) в БД.
5. **iOS App**: запрашивает результат по `taskId` и отображает координаты / UI детекции.

Отдельный тулчейн в `MLService/` (датасет → ROI → сегментация U-Net) описан в [MLService/README.md](MLService/README.md) и не обязан совпадать с каждым шагом продакшен-запроса через Backend.

## Функциональность

### Реализовано

✅ Backend API (Vapor)
- Загрузка DICOM файлов
- Управление задачами анализа
- Хранение результатов в SQLite
- HTTP клиент для ML Service

✅ ML Service (Python)
- Парсинг DICOM и сбор объёма
- Детекция TMJ (3D CNN; см. эксперименты и `MODEL_PATH`)
- Инструменты обучения, датасета и сегментации (U-Net) в репозитории
- Поддержка MPS / CUDA по конфигурации

### В разработке / улучшения

🔄 Качество детекции (метрики на валидации, см. [QUICKSTART.md](QUICKSTART.md))
🔄 Расширение UI (3D-визуализация, экспорт результатов и т.д.)

## Структура проекта

```
MasterProject/
├── Backend/
├── MLService/
├── iOSApp/MasterDoctor/       # Xcode-проект и модули
├── docs/                      # Оглавление и контекст (в т.ч. для агентов)
├── examples/
├── README.md
├── AGENTS.md
├── QUICKSTART.md
└── TMJ_DETECTION_SETUP.md
```

## Разработка

### Backend (Swift)

Редактирование и запуск:
```bash
cd Backend
swift run App
```

### ML Service (Python)

Запуск с hot-reload:
```bash
cd MLService
source venv/bin/activate
uvicorn app:app --reload --port 8001
```

## Тестирование

Проверка сервисов:

```bash
curl http://localhost:8080/health
curl http://localhost:8001/health
```

Интеграционная загрузка DICOM — через **multipart** на `POST /api/analysis` (как iOS-клиент или Postman). Подробности и типичные проблемы: [TMJ_DETECTION_SETUP.md](TMJ_DETECTION_SETUP.md).

## Конфигурация

### Backend

Переменные окружения:
- `ML_SERVICE_URL` - URL ML сервиса (по умолчанию: `http://localhost:8001`)

### ML Service

Переменные окружения:
- `MODEL_PATH` - путь к файлу модели (по умолчанию: `models/segmentation_model.pth`)

## Документация

- [Оглавление docs/](docs/README.md)
- [Backend README](Backend/README.md)
- [ML Service README](MLService/README.md)
- [Быстрый старт детекции](QUICKSTART.md)
- [Настройка TMJ pipeline](TMJ_DETECTION_SETUP.md)
- [Публичная CBCT-когорта: метки и сбор датасета](docs/cbct-public-cohort-dataset.md)

Черновики планов в каталоге `.cursor/plans/` при работе в Cursor могут быть только локально (каталог `.cursor/` не в git).

## Лицензия

Учебный проект для магистратуры

## Автор

Магистерская программа "ИИ в медицине"

