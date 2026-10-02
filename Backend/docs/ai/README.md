# Инженерные свидетельства Backend

Дата: 2026-10-02. Источник — отслеживаемые исходники репозитория. Сверка статическая: сервисы, миграции, сборки и тесты не запускались.
Нормативные требования и TC находятся в [spec](../spec/README.md).
Описанные ниже ограничения реализации не легализуют дефекты.

## Источники и поток

| Свидетельство | Исходник |
|---|---|
| Регистрация обработчиков | [routes.swift](../../Sources/App/routes.swift) |
| Загрузка, фоновая работа, выдача результата | [AnalysisController.swift](../../Sources/App/Controllers/AnalysisController.swift) |
| Запись и удаление файлов | [DICOMStorageService.swift](../../Sources/App/Services/DICOMStorageService.swift) |
| Запрос и декодирование ML | [MLServiceClient.swift](../../Sources/App/Services/MLServiceClient.swift) |
| Задача и её миграция | [AnalysisTask.swift](../../Sources/App/Models/AnalysisTask.swift) |
| Результат и его миграция | [AnalysisResult.swift](../../Sources/App/Models/AnalysisResult.swift) |
| SQLite, JSON, лимиты и автомиграция | [configure.swift](../../Sources/App/configure.swift) |
| Проверка живости | [HealthController.swift](../../Sources/App/Controllers/HealthController.swift) |

Порядок загрузки: декодирование → проверка непустого массива → UUID → сохранение каталога серии → запись задачи `pending` → `Task.detached` → возврат идентификатора. Фоновая работа: `processing` → запрос ML → при наличии `error_message` состояние `failed`; иначе сериализация локализации → запись результата → `completed`.
Ошибка фонового процесса логируется; запись `failed` в outer catch использует `try?`.
Ответ загрузки не гарантирует, что клиент успеет увидеть `pending`.

<a id="upload"></a>
## Загрузка и маршруты

| Метод / ресурс | Наблюдаемая реализация |
|---|---|
| `POST /api/analysis` | multipart декодируется в локальный DTO `files: [File]`; JSON-ответ `task_id` с UUID |
| `GET /api/analysis/{taskId}` | Статус и optional-результат в одном ответе |
| `GET /health` | `status="ok"`, `service="vapor-backend"`, `timestamp`; зависимости не проверяются |

Маршрут загрузки собирает тело до `100mb`; общий default в configure — `500mb`. Пустой декодированный массив даёт `400 No files provided`. Неверный UUID при GET — `400 Invalid task ID`; неизвестный — `404 Task not found`.
Отдельного `/api/analysis/{taskId}/status` и маршрутов списка/отмены/удаления нет.
Повторяющиеся multipart-части iOS требуют runtime-проверки совместимости с `[File]`.

## Представление результата

Backend JSON использует `convertToSnakeCase`. GET выдаёт `task_id`, `status`, optional `error_message`, `tmj_left`, `tmj_right`, `volume_shape`, `created_at`, `updated_at`. Даты GET строятся `ISO8601DateFormatter`; optional-поля могут отсутствовать, точная сериализация не проверена. Статусы: `pending`, `processing`, `completed`, `failed`. Локализация хранится и передаётся как JSON-строка внутри JSON ответа: `{"center":[...],"bbox":[...]}`. `center` — float `[z,y,x]`, `bbox` — integer `[z1,y1,x1,z2,y2,x2]`, `volume_shape` — integer `[depth,height,width]`. Оси сверены с [ML app.py](../../../MLService/app.py); Backend не переориентирует объём. Длины, границы координат и положительность размеров Backend не проверяет.
GET выбирает первую запись результата по task ID; единственность не гарантирована схемой.
Legacy-поля `slices_data`, `masks_data`, `parameters`, `diagnosis` этим API не выдаются.

## Транспорт к ML

Адрес: `ML_SERVICE_URL`, default `http://localhost:8001`; запрос `POST /process`. Тело содержит `task_id` и повторяющиеся `files` с типом `application/dicom`. Из каталога берутся только файлы с case-insensitive расширением `.dcm`; порядок перечисления не фиксирован; нечитаемые файлы пропускаются. Отсутствие `.dcm` бросает `No DICOM files found in directory`. Таймаут обращения — 10 минут. Успех транспорта — только HTTP 200; error body собирается до 1 MiB, success body — до 50 MiB; затем декодируется DTO. ML DTO требует `task_id` и `status`; optional `tmj` при наличии требует обе стороны. Преобразование в `MLProcessingResult` теряет `task_id` и `status`.
Присутствующий `error_message`, включая пустую строку, означает `failed`;
его отсутствие позволяет сохранить результат без локализации и выставить `completed`.

## Хранение и риски

SQLite: `db.sqlite`; миграции вызываются при configure. Серия: `uploads/{UUID}/` относительно working directory; путь записан в задаче. `saveSeries` добавляет исходное имя напрямую, пропускает недоступные bytes, подавляет ошибки записи через `try?`; содержимое DICOM не валидируется. `saveFile` — отдельный helper; маршрут серии использует `saveSeries`. Delete helpers есть, но путь анализа их не вызывает; uploads автоматически не очищаются. Запись задачи, файлов, результата и статуса не составляет одну транзакцию.
Схема связывает результат с задачей без уникального ограничения на `task_id`.
Авторизация и изоляция пользователей не зарегистрированы в маршрутах.

## Границы совместимости и проверок

[AnalysisEndpoint](../../../iOSApp/MasterDoctor/MainFeatures/Sources/AnalyticsApp/Services/Endpoint/AnalysisEndpoint.swift) запрашивает `analysis/{id}/status`; этот путь у Backend отсутствует. [TMJDetectionFetchService](../../../iOSApp/MasterDoctor/MainFeatures/Sources/AnalyticsApp/Services/TMJDetectionFetchService.swift) делает до 10 запросов статуса с паузами 2 секунды в незавершённой ветке; сетевое время добавляется к паузам. Это не общий дедлайн в 20 секунд. [Package.swift](../../Package.swift) не объявляет test target; отслеживаемых Backend-тестов нет. [CI](../../../.github/workflows/ci.yml) содержит `swift build -v`, без Backend-тестов.
Проверки приёмки отсутствуют; статическая сверка не подтверждает работоспособность
SQLite, декодирования multipart, HTTP и сквозной обработки конкретной серии.
