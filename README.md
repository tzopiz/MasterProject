# AI Doctor — анализ ВНЧС

Магистерский проект: iOS-приложение с ИИ для анализа височно-нижнечелюстного сустава (ВНЧС / TMJ) по КЛКТ (CBCT).

**Документация:** [AGENTS.md](AGENTS.md) (правила для агентов и оформления доков) · [MLService/README.md](MLService/README.md) (ML, карта папок) · [.github/README.md](.github/README.md) (CI)

## Назначение и поток данных

Сценарий: загрузка **серии DICOM**, вызов ML, возврат координат / области для клиента.

```text
iOS (SwiftUI)  --HTTP-->  Backend (Vapor + SQLite)  --HTTP-->  ML Service (FastAPI + PyTorch)
```

1. Клиент отправляет на Backend серию DICOM (multipart).
2. Backend сохраняет файлы, создаёт задачу, вызывает ML Service.
3. ML строит объём, детекция TMJ, возвращает координаты и метаданные (в т.ч. размер объёма).
4. Backend сохраняет результат в БД; клиент получает ответ по `taskId`.

## Глоссарий

| Термин | Пояснение |
|--------|------------|
| ВНЧС / TMJ | Височно-нижнечелюстной сустав |
| КЛКТ / CBCT | Конусно-лучевая КТ |
| DICOM | Медицинский формат; в проекте — серии `.dcm` |
| Детектор | Локализация суставов в 3D (центр, bbox) |
| ROI | Область интереса; разметка и кропы |

## Архитектура репозитория

1. **Backend** (Swift Vapor) — API задач и данные
2. **MLService** (Python FastAPI) — инференс и обработка DICOM
3. **iOS** (SwiftUI) — клиент: [iOSApp/README.md](iOSApp/README.md)

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

## Стек

- **Backend:** Swift 5.9+, Vapor 4, Fluent + SQLite, AsyncHTTPClient
- **ML:** Python 3.9+, FastAPI, PyTorch, pydicom, scikit-image, OpenCV
- **iOS:** Swift 5.9+, SwiftUI; URL API — в коде (`AnalysisEndpoint` и т.п.)

## Быстрый старт (локальная цепочка)

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

Сервис: **http://127.0.0.1:8001** (или порт из лога). Веса: **`MODEL_PATH`**, иначе последний `experiments/detector_*/best_model.pth`, иначе **`models/tmj_detector_best.pth`** (см. `app.py`).

```bash
curl http://127.0.0.1:8001/health
```

### 2. Backend

```bash
cd Backend
swift build
swift run App
```

По умолчанию **http://127.0.0.1:8080**; `curl http://127.0.0.1:8080/health`. Если ML на другом хосте:

```bash
export ML_SERVICE_URL="http://127.0.0.1:8001"
swift run App
```

### 3. iOS

Откройте `iOSApp/MasterDoctor/MasterDoctor.xcodeproj`. Для загрузки папки с `.dcm` проверьте права в **Info.plist** и использование документ-пикера / `.fileImporter` в текущей сборке.

## HTTP API (сверяйтесь с кодом)

### Backend (`Backend/Sources/App/Controllers/`)

| Метод | Путь | Назначение |
|-------|------|------------|
| GET | `/health` | Живость сервиса |
| POST | `/api/analysis` | Multipart: серия DICOM (`SeriesUploadRequest`, поле `files`) |
| GET | `/api/analysis/{taskId}` | Статус и результат в одном ответе (отдельного только-status маршрута может не быть) |

### ML Service (`MLService/app.py`)

| Метод | Путь |
|-------|------|
| GET | `/health` |
| GET | `/models/status` |
| POST | `/process` |

Ответ по задаче может содержать **`tmjLeft` / `tmjRight`** (JSON-строки с центром/bbox) и **`volumeShape`** — см. `AnalysisResponse` в Swift.

## Устранение неполадок

- **ML не находит вес:** `ls experiments/detector_*/best_model.pth` или положите файл в `MLService/models/tmj_detector_best.pth` ([MLService/models/README.md](MLService/models/README.md)).
- **Backend не достучится до ML:** `curl` на `:8001/health`, переменная `ML_SERVICE_URL`.
- **iOS не достучится до Backend:** URL в `AnalysisEndpoint.swift`; для устройства — IP машины вместо `localhost`.
- **Мониторинг обучения:** логи в терминале, где запущен `train_detector.py` / `train_*.py`; артефакты в `experiments/<прогон>/` (`train.log`, `metrics.jsonl`, `best_model.pth`). Смотреть хвост: `tail -f experiments/<прогон>/train.log` (если лог пишется в файл).

## Конфигурация

| Переменная | Где | Назначение |
|------------|-----|------------|
| `ML_SERVICE_URL` | Backend | URL ML (часто `http://localhost:8001`) |
| `MODEL_PATH` | MLService | Путь к весам детектора |

## Структура каталогов (верхний уровень)

```text
MasterProject/
├── Backend/
├── MLService/
├── iOSApp/
├── README.md
└── AGENTS.md
```

Тулчейн обучения и когорта: [MLService/README.md](MLService/README.md). Данные ML: [MLService/data/README.md](MLService/data/README.md).

## Разработка

```bash
cd Backend && swift run App
cd MLService && source venv/bin/activate && uvicorn app:app --reload --port 8001
```

## Лицензия и автор

Учебный проект (магистратура «ИИ в медицине»).
