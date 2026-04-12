# AI Doctor - Анализ ВНЧС

Магистерский проект: iOS приложение с ИИ для анализа височно-нижнечелюстного сустава (ВНЧС) по снимкам КЛКТ.

**Документация:** [AGENTS.md](AGENTS.md) (для ИИ-агентов) · [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) (термины, API, структура) · [TMJ_DETECTION_SETUP.md](TMJ_DETECTION_SETUP.md) (типичные проблемы пайплайна)

## Архитектура

Проект состоит из трёх компонентов:

1. **Backend** (Swift Vapor) — API для задач и данных
2. **MLService** (Python FastAPI) — ML-инференс и обработка DICOM
3. **iOS App** (Swift/SwiftUI) — клиент ([iOSApp/ModularAppArchitecture.md](iOSApp/ModularAppArchitecture.md))

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   iOS App   │ ◄─────► │    Backend   │ ◄─────► │  ML Service │
│  (Swift)    │   API   │   (Vapor)    │   API   │  (FastAPI)  │
└─────────────┘         └──────────────┘         └─────────────┘
                              │
                              ▼
                        ┌──────────┐
                        │ SQLite   │
                        └──────────┘
```

## Технологический стек

### Backend (Vapor)

- Swift 5.9+, Vapor 4.x, Fluent + SQLite, AsyncHTTPClient

### ML Service

- Python 3.9+, FastAPI, PyTorch, pydicom, scikit-image, OpenCV

### iOS App

- Swift 5.9+, SwiftUI; базовый URL API — в коде клиента (`AnalysisEndpoint` и т.п.)

## Быстрый старт (локальная демо-цепочка)

### Требования

- Swift 5.9+ и Xcode (Backend + iOS)
- Python 3.9+ (MLService)
- macOS 13+ или Linux для Backend/ML

### 1. ML Service

```bash
cd MLService
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Сервис слушает **http://127.0.0.1:8001** (или порт из лога). Вес детектора: переменная **`MODEL_PATH`**, иначе поиск последнего `experiments/detector_*/best_model.pth`, иначе ожидается **`models/tmj_detector_best.pth`** (см. `app.py`).

Пример с явным чекпойнтом:

```bash
export MODEL_PATH="experiments/detector_20251124_175805/best_model.pth"
python app.py
```

Проверка:

```bash
curl http://127.0.0.1:8001/health
```

### 2. Backend

```bash
cd Backend
swift build
swift run App
```

По умолчанию **http://127.0.0.1:8080**. Проверка:

```bash
curl http://127.0.0.1:8080/health
```

Если ML на другом хосте/порту:

```bash
export ML_SERVICE_URL="http://127.0.0.1:8001"
swift run App
```

### 3. iOS

1. Открыть `iOSApp/MasterDoctor/MasterDoctor.xcodeproj` в Xcode
2. Запустить схему на симуляторе или устройстве
3. Выбрать папку с `.dcm`, запустить сценарий анализа (экран зависит от текущей сборки; см. `TMJ_DETECTION_SETUP.md`)

### Устранение неполадок

- **ML не находит вес:** проверьте `ls experiments/detector_*/best_model.pth` или положите релизный файл в `MLService/models/tmj_detector_best.pth` ([MLService/models/README.md](MLService/models/README.md)).
- **Backend не достучится до ML:** `curl` на `:8001/health`, выставьте `ML_SERVICE_URL`.
- **iOS и папка DICOM:** права в `Info.plist`, использование `.fileImporter` / документ-пикера — см. `TMJ_DETECTION_SETUP.md`.

## API (кратко)

Контракты уточняйте в коде: `Backend/Sources/App/Controllers/`, `MLService/app.py`.

### Backend

- `GET /health`
- `POST /api/analysis` — multipart, серия DICOM (`AnalysisController`, `SeriesUploadRequest`)
- `GET /api/analysis/{taskId}` — статус и результат

### ML Service

- `GET /health`, `GET /models/status`, `POST /process`

Ответ анализа по `taskId` может содержать поля вроде **`tmjLeft` / `tmjRight`** (JSON-строки с координатами/bbox) и **`volumeShape`** — см. `AnalysisResponse` в Swift.

## Workflow

1. iOS отправляет серию `.dcm` на Backend (`POST /api/analysis`).
2. Backend сохраняет файлы, создаёт задачу, вызывает ML Service.
3. ML строит объём, детекция TMJ, возвращает координаты/метаданные.
4. Backend пишет результат в SQLite; клиент опрашивает по `taskId`.

Тулчейн обучения/датасета в `MLService/` описан в [MLService/README.md](MLService/README.md).

## Функциональность

### Реализовано

- Backend API (загрузка DICOM, задачи, SQLite, клиент к ML)
- ML Service: парсинг DICOM, детекция TMJ, MPS/CUDA по окружению

### В разработке / улучшения

- Качество детекции и UX (см. [TMJ_DETECTION_SETUP.md](TMJ_DETECTION_SETUP.md))
- Расширение UI (3D, экспорт и т.д.)

## Структура репозитория

```
MasterProject/
├── Backend/
├── MLService/
├── iOSApp/MasterDoctor/
├── README.md
├── PROJECT_CONTEXT.md
├── TMJ_DETECTION_SETUP.md
└── AGENTS.md
```

## Разработка

### Backend

```bash
cd Backend
swift run App
```

### ML Service (reload)

```bash
cd MLService
source venv/bin/activate
uvicorn app:app --reload --port 8001
```

## Тестирование

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8001/health
```

Интеграция через multipart на `POST /api/analysis` — подробности в [TMJ_DETECTION_SETUP.md](TMJ_DETECTION_SETUP.md).

## Конфигурация

| Переменная | Где | Назначение |
|------------|-----|------------|
| `ML_SERVICE_URL` | Backend | URL ML Service (по умолчанию `http://localhost:8001`) |
| `MODEL_PATH` | MLService | Путь к весам детектора |

## Лицензия

Учебный проект для магистратуры.

## Автор

Магистерская программа «ИИ в медицине».
