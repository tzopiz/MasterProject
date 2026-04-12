# Backend Service

Swift Vapor backend для обработки DICOM файлов и управления задачами анализа ВНЧС.

## Структура проекта

```
Backend/
├── Sources/App/
│   ├── Controllers/         # API контроллеры
│   ├── Models/             # Модели базы данных
│   ├── Services/           # Бизнес-логика
│   ├── configure.swift     # Конфигурация приложения
│   ├── routes.swift        # Маршруты
│   └── entrypoint.swift    # Точка входа
└── Package.swift           # Зависимости
```

## Требования

- Swift 5.9+
- macOS 13+

## Установка и запуск

1. Установить зависимости:
```bash
cd Backend
swift package resolve
```

2. Собрать проект:
```bash
swift build
```

3. Запустить сервер:
```bash
swift run App
```

Сервер запустится на `http://localhost:8080`

## API Endpoints

### Health Check
```
GET /health
```

Возвращает статус сервиса.

### Загрузка серии DICOM
```
POST /api/analysis
Content-Type: multipart/form-data
```

Тело: несколько файлов в частях формы; на сервере ожидается декодирование в `SeriesUploadRequest` с массивом `files` (см. `AnalysisController.swift`).

Ответ (JSON, ключи в snake_case):
```json
{
  "task_id": "uuid"
}
```

### Результат и статус задачи
```
GET /api/analysis/{taskId}
```

В одном ответе: `status` задачи (`pending`, `processing`, `completed`, `failed`), при готовности — поля с результатами TMJ и `volume_shape`. Отдельного маршрута только для статуса в текущей реализации нет.

## Конфигурация

Переменные окружения:

- `ML_SERVICE_URL` - URL ML сервиса (по умолчанию: `http://localhost:8001`)

## База данных

Используется SQLite для разработки. Файл базы данных: `db.sqlite`

### Таблицы:

- `analysis_tasks` - задачи анализа
- `analysis_results` - результаты анализа

## См. также

- [README.md](../README.md) — обзор репозитория и быстрый старт
- [AGENTS.md](../AGENTS.md) — правила для агентов и документации
- [iOSApp/README.md](../iOSApp/README.md) — клиент

