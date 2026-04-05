# Сведения для агентов (AI / автоматизация)

Краткий вход в репозиторий **MasterProject** (AI Doctor — анализ ВНЧС по КЛКТ).

## С чего начать

1. **[docs/README.md](docs/README.md)** — оглавление всей документации и сценарии «что читать».
2. **[docs/project-context.md](docs/project-context.md)** — термины, поток данных, карта каталогов, актуальные HTTP-маршруты (проверять по коду при сомнениях).
3. Корневой **[README.md](README.md)** — обзор, быстрый старт, ссылки на сервисы.

## Границы системы

| Компонент | Путь | Роль |
|-----------|------|------|
| Backend | `Backend/` | Vapor, SQLite, приём DICOM-серии, вызов ML, выдача результатов |
| ML Service | `MLService/` | FastAPI, PyTorch, детекция/обработка объёма |
| iOS | `iOSApp/MasterDoctor/` | SwiftUI-клиент |

Доменные синонимы в документах: **TMJ** = **ВНЧС**, **CBCT** = **КЛКТ**.

## Правило актуальности

При расхождении текста с кодом **источник истины — исходники** (`AnalysisController.swift`, `routes.swift`, `app.py`, эндпоинты в клиенте). В [docs/project-context.md](docs/project-context.md) зафиксированы известные нюансы.

## Частые задачи

| Задача | Куда смотреть |
|--------|----------------|
| Запуск всей цепочки | [QUICKSTART.md](QUICKSTART.md), [TMJ_DETECTION_SETUP.md](TMJ_DETECTION_SETUP.md) |
| ML: обучение, датасет, инструменты | [MLService/README.md](MLService/README.md), [MLService/TRAINING_MONITORING.md](MLService/TRAINING_MONITORING.md) |
| Классификатор положения ВНЧС (Colab / DataSphere, v3–v5) | [MLService/google_colab/README.md](MLService/google_colab/README.md), [MLService/google_colab/POSITION_CLASSIFIER_EXPERIMENTS.md](MLService/google_colab/POSITION_CLASSIFIER_EXPERIMENTS.md) |
| Публичная CBCT-когорта (DOCX→метки, Яндекс.Диск, zip, датасет) | [docs/cbct-public-cohort-dataset.md](docs/cbct-public-cohort-dataset.md) |
| API Backend | `Backend/Sources/App/Controllers/`, [Backend/README.md](Backend/README.md) |
| Сеть iOS ↔ Backend | `iOSApp/.../AnalysisEndpoint.swift` (базовый URL и пути) |
| Презентация / экспорт HTML→PDF | [docs/presentation/EXPORT_README.md](docs/presentation/EXPORT_README.md) |
| Краткий питч проекта | [docs/presentation/PROJECT_PRESENTATION.md](docs/presentation/PROJECT_PRESENTATION.md) |

## Репозиторий

- В **git** не попадают личные и сгенерированные файлы (см. `.gitignore`: `diplom.pdf`, `docs/presentation/presentation.pdf`, часть `.docx` в `docs/presentation/` и т.д.).
- Каталог `.cursor/` в репозитории не версионируется; локальные планы Cursor при клонировании могут отсутствовать.
