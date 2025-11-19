# AI Doctor - Анализ ВНЧС

Магистерский проект: iOS приложение с ИИ для анализа височно-нижнечелюстного сустава (ВНЧС) по снимкам КЛКТ.

## Архитектура

Проект состоит из трех компонентов:

1. **Backend** (Swift Vapor) - API для управления задачами и данными
2. **MLService** (Python FastAPI) - ML инференс и обработка DICOM
3. **iOS App** (Swift/SwiftUI) - клиентское приложение (в разработке)

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

### iOS App (будущее)
- Swift 5.9+
- SwiftUI - UI фреймворк
- URLSession - сетевые запросы

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

### Backend Endpoints

- `GET /health` - health check
- `POST /api/dicom/upload` - загрузка DICOM файла
- `GET /api/analysis/{taskId}` - получить результаты анализа
- `GET /api/analysis/{taskId}/status` - статус задачи

### ML Service Endpoints

- `GET /health` - health check
- `GET /models/status` - статус модели
- `POST /process` - обработка DICOM файла

## Workflow

1. **Загрузка DICOM**: Пользователь загружает .dcm файл через iOS приложение
2. **Backend**: Сохраняет файл, создает задачу в БД, отправляет в ML Service
3. **ML Service**:
   - Парсит DICOM файл
   - Находит нужные срезы (ортогональные, сагиттальные, фронтальные)
   - Выполняет сегментацию ВНЧС
   - Вычисляет геометрические параметры
   - Генерирует диагноз и рекомендации
4. **Backend**: Сохраняет результаты в БД
5. **iOS App**: Получает и отображает результаты

## Функциональность

### Реализовано

✅ Backend API (Vapor)
- Загрузка DICOM файлов
- Управление задачами анализа
- Хранение результатов в SQLite
- HTTP клиент для ML Service

✅ ML Service (Python)
- Парсинг DICOM файлов
- Поиск релевантных срезов
- Архитектура модели сегментации (U-Net)
- Вычисление геометрических параметров
- Логика диагностики
- Dummy mode (без обученной модели)

### В разработке

🔄 ML модель
- Обучение модели сегментации на реальных данных
- Оптимизация инференса

🔄 iOS приложение
- UI/UX дизайн
- DICOM viewer
- Отображение результатов
- Чат с ИИ

## Структура проекта

```
MasterProject/
├── Backend/                    # Swift Vapor backend
│   ├── Sources/App/
│   │   ├── Controllers/       # API контроллеры
│   │   ├── Models/            # Модели БД
│   │   ├── Services/          # Бизнес-логика
│   │   ├── configure.swift
│   │   ├── routes.swift
│   │   └── entrypoint.swift
│   └── Package.swift
│
├── MLService/                  # Python ML сервис
│   ├── app.py                 # FastAPI приложение
│   ├── models/                # ML модели
│   ├── services/              # Обработка данных
│   ├── utils/                 # Утилиты
│   └── requirements.txt
│
└── README.md                  # Этот файл
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

### Загрузка тестового DICOM файла

```bash
# Требуется DICOM файл для тестирования
curl -X POST http://localhost:8080/api/dicom/upload \
  -H "Content-Type: application/json" \
  -d '{"filename": "test.dcm", "data": "..."}'
```

### Проверка статуса задачи

```bash
curl http://localhost:8080/api/analysis/{taskId}/status
```

## Конфигурация

### Backend

Переменные окружения:
- `ML_SERVICE_URL` - URL ML сервиса (по умолчанию: `http://localhost:8001`)

### ML Service

Переменные окружения:
- `MODEL_PATH` - путь к файлу модели (по умолчанию: `models/segmentation_model.pth`)

## Документация

- [Backend README](Backend/README.md)
- [ML Service README](MLService/README.md)
- [Architecture Plan](.cursor/plans/architecture-reference.md)
- [Backend Implementation Plan](.cursor/plans/backend-ml-implementation.md)

## Лицензия

Учебный проект для магистратуры

## Автор

Магистерская программа "ИИ в медицине"

