# Руководство по развертыванию

Инструкции по развертыванию Backend и ML Service на различных платформах.

## Локальная разработка

### Требования
- macOS 13+ или Linux
- Swift 5.9+ (для Backend)
- Python 3.9+ (для ML Service)

### Быстрый старт

1. **Backend**
```bash
cd Backend
swift build
swift run App
```

2. **ML Service**
```bash
cd MLService
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

3. **С использованием скриптов**
```bash
# Terminal 1
./Backend/start.sh

# Terminal 2
./MLService/start.sh
```

## Деплой на бесплатные хостинги

### 1. Render.com (Рекомендуется)

#### Backend (Vapor)

1. Создайте `Dockerfile` в папке `Backend/`:
```dockerfile
FROM swift:5.9-focal as build

WORKDIR /build
COPY . .

RUN swift build -c release --static-swift-stdlib

FROM ubuntu:focal
RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true \
    && apt-get -q update \
    && apt-get -q dist-upgrade -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /build/.build/release/App /app

EXPOSE 8080

ENTRYPOINT ["./App"]
CMD ["serve", "--env", "production", "--hostname", "0.0.0.0", "--port", "8080"]
```

2. Зарегистрируйтесь на [Render.com](https://render.com)
3. Создайте новый Web Service
4. Подключите GitHub репозиторий
5. Настройки:
   - Build Command: `docker build -t backend ./Backend`
   - Start Command: Используется из Dockerfile
   - Environment Variables:
     - `ML_SERVICE_URL`: URL ML Service на Render

#### ML Service (Python)

1. Создайте `Dockerfile` в папке `MLService/`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установить зависимости системы
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копировать requirements и установить зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копировать код приложения
COPY . .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

2. На Render.com создайте новый Web Service
3. Настройки:
   - Build Command: `docker build -t mlservice ./MLService`
   - Start Command: Используется из Dockerfile
   - Environment Variables:
     - `MODEL_PATH`: `models/segmentation_model.pth` (опционально)

### 2. Railway.app

#### Подготовка

1. Установите Railway CLI:
```bash
npm install -g @railway/cli
```

2. Войдите в аккаунт:
```bash
railway login
```

#### Backend

```bash
cd Backend
railway init
railway up
```

Добавьте переменную окружения в Railway Dashboard:
- `ML_SERVICE_URL`: URL вашего ML Service

#### ML Service

```bash
cd MLService
railway init
railway up
```

### 3. Fly.io

#### Backend

1. Установите Fly CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

2. Создайте приложение:
```bash
cd Backend
fly launch
```

3. Настройте `fly.toml`:
```toml
app = "your-backend-app"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8080"

[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

4. Деплой:
```bash
fly deploy
```

#### ML Service

```bash
cd MLService
fly launch
fly deploy
```

## Конфигурация для продакшена

### Backend

В `Backend/Sources/App/configure.swift` добавьте проверку окружения:

```swift
public func configure(_ app: Application) async throws {
    // Production settings
    if app.environment == .production {
        app.routes.defaultMaxBodySize = "1gb"
        app.http.server.configuration.port = 8080
    }
    
    // ... rest of configuration
}
```

### ML Service

Создайте `MLService/config.py`:

```python
import os

class Config:
    # Production settings
    IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"
    
    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "models/segmentation_model.pth")
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = int(os.getenv("PORT", "8001"))
    
    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Limits
    MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
    
    # Timeouts
    PROCESSING_TIMEOUT = 600  # 10 minutes
```

## База данных для продакшена

### Переход на PostgreSQL

1. Обновите `Backend/Package.swift`:
```swift
.package(url: "https://github.com/vapor/fluent-postgres-driver.git", from: "2.8.0"),
```

2. Обновите `configure.swift`:
```swift
if app.environment == .production {
    // PostgreSQL for production
    let hostname = Environment.get("DATABASE_HOST") ?? "localhost"
    let username = Environment.get("DATABASE_USER") ?? "vapor"
    let password = Environment.get("DATABASE_PASSWORD") ?? ""
    let database = Environment.get("DATABASE_NAME") ?? "vapor"
    
    app.databases.use(.postgres(
        hostname: hostname,
        username: username,
        password: password,
        database: database
    ), as: .psql)
} else {
    // SQLite for development
    app.databases.use(.sqlite(.file("db.sqlite")), as: .sqlite)
}
```

3. На Render/Railway/Fly добавьте PostgreSQL addon и настройте переменные окружения.

## Мониторинг

### Логирование

Backend автоматически логирует в stdout. Для продакшена можно добавить:

```swift
import Logging

// В configure.swift
if app.environment == .production {
    LoggingSystem.bootstrap { label in
        var handler = StreamLogHandler.standardOutput(label: label)
        handler.logLevel = .info
        return handler
    }
}
```

ML Service уже настроен на логирование через `logging` модуль Python.

### Health checks

Оба сервиса имеют `/health` endpoints для health checks.

Настройте проверки на платформе хостинга:
- Path: `/health`
- Interval: 30 seconds
- Timeout: 5 seconds
- Unhealthy threshold: 3 failures

## Хранилище файлов

Для продакшена рекомендуется использовать облачное хранилище вместо локального:

### AWS S3

```swift
// В Backend
import SotoS3

// Загрузка в S3 вместо локального диска
func uploadToS3(_ data: ByteBuffer, key: String) async throws -> String {
    let client = AWSClient()
    let s3 = S3(client: client)
    
    let request = S3.PutObjectRequest(
        bucket: "your-bucket",
        key: key,
        body: .byteBuffer(data)
    )
    
    _ = try await s3.putObject(request)
    return key
}
```

## Масштабирование

### Горизонтальное масштабирование

Backend и ML Service stateless, можно запускать несколько инстансов:

```bash
# На Render/Railway/Fly настройте autoscaling
# Или вручную увеличьте количество инстансов
```

### Очередь задач

Для большой нагрузки рекомендуется добавить очередь (Redis, RabbitMQ):

```swift
// Backend: добавить задачу в очередь вместо синхронного вызова ML Service
import QueuesRedisDriver

app.queues.use(.redis(url: redisURL))

// Создать джобу
struct ProcessDICOMJob: AsyncJob {
    func dequeue(_ context: QueueContext, _ payload: ProcessDICOMPayload) async throws {
        // Вызвать ML Service
    }
}
```

## SSL/TLS

Большинство платформ (Render, Railway, Fly) автоматически предоставляют SSL сертификаты.

Для собственного сервера используйте Let's Encrypt:

```bash
# Nginx reverse proxy с SSL
sudo apt-get install nginx certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Backup

### База данных

Автоматический backup PostgreSQL на Render/Railway/Fly обычно включен.

Для ручного backup:

```bash
pg_dump $DATABASE_URL > backup.sql
```

### Файлы моделей

```bash
# Backup моделей в S3
aws s3 cp models/ s3://your-bucket/models/ --recursive
```

## Безопасность

### Rate limiting

```swift
// В Backend добавьте middleware для rate limiting
import Vapor

struct RateLimitMiddleware: AsyncMiddleware {
    func respond(to request: Request, chainingTo next: AsyncResponder) async throws -> Response {
        // Проверить количество запросов от IP
        // Ограничить если превышен лимит
        return try await next.respond(to: request)
    }
}

app.middleware.use(RateLimitMiddleware())
```

### Валидация входных данных

Уже реализована в контроллерах. Дополнительно можно добавить:

```swift
// Проверка размера файла
guard file.data.readableBytes <= 1024 * 1024 * 1024 else {
    throw Abort(.payloadTooLarge)
}

// Проверка MIME типа
guard file.contentType == "application/dicom" else {
    throw Abort(.unsupportedMediaType)
}
```

## Стоимость

### Бесплатные тарифы (примерная)

- **Render**: 750 часов/месяц бесплатно
- **Railway**: $5 кредит/месяц
- **Fly.io**: 3 VM бесплатно (ограниченные ресурсы)

### Оценка стоимости для масштабирования

При 1000 пользователях/месяц:
- Backend: ~$7-15/месяц
- ML Service: ~$25-50/месяц (зависит от размера модели)
- База данных: ~$7/месяц (PostgreSQL)
- Хранилище: ~$5/месяц (S3)

**Итого**: ~$44-77/месяц

