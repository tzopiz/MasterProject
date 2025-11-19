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

### Загрузка DICOM файла
```
POST /api/dicom/upload
Content-Type: application/json

{
  "filename": "scan.dcm",
  "data": <ByteBuffer>
}
```

Возвращает:
```json
{
  "taskId": "uuid",
  "status": "uploaded",
  "message": "File uploaded successfully. Processing started."
}
```

### Получить результаты анализа
```
GET /api/analysis/{taskId}
```

Возвращает результаты обработки DICOM файла.

### Получить статус задачи
```
GET /api/analysis/{taskId}/status
```

Возвращает текущий статус задачи (pending, processing, completed, failed).

## Конфигурация

Переменные окружения:

- `ML_SERVICE_URL` - URL ML сервиса (по умолчанию: `http://localhost:8001`)

## База данных

Используется SQLite для разработки. Файл базы данных: `db.sqlite`

### Таблицы:

- `analysis_tasks` - задачи анализа
- `analysis_results` - результаты анализа

