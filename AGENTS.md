# Сведения для агентов (AI / автоматизация)

Краткий вход в репозиторий **MasterProject** (AI Doctor — анализ ВНЧС по КЛКТ).

## С чего начать

1. **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** — термины, поток данных, карта каталогов, актуальные HTTP-маршруты (проверять по коду при сомнениях).
2. Корневой **[README.md](README.md)** — обзор, быстрый старт, ссылки на сервисы.

## Границы системы

| Компонент | Путь | Роль |
|-----------|------|------|
| Backend | `Backend/` | Vapor, SQLite, приём DICOM-серии, вызов ML, выдача результатов |
| ML Service | `MLService/` | FastAPI, PyTorch, детекция/обработка объёма |
| iOS | `iOSApp/MasterDoctor/` | SwiftUI-клиент |

Доменные синонимы в документах: **TMJ** = **ВНЧС**, **CBCT** = **КЛКТ**.

## Правило актуальности

При расхождении текста с кодом **источник истины — исходники** (`AnalysisController.swift`, `routes.swift`, `app.py`, эндпоинты в клиенте). В [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) зафиксированы известные нюансы.

## Частые задачи

| Задача | Куда смотреть |
|--------|----------------|
| Запуск всей цепочки | [QUICKSTART.md](QUICKSTART.md), [TMJ_DETECTION_SETUP.md](TMJ_DETECTION_SETUP.md) |
| ML: обучение, датасет, инструменты | [MLService/README.md](MLService/README.md), [MLService/TRAINING_MONITORING.md](MLService/TRAINING_MONITORING.md) |
| Классификатор положения ВНЧС (Colab / DataSphere, v3–v5) | [MLService/google_colab/README.md](MLService/google_colab/README.md), [MLService/google_colab/POSITION_CLASSIFIER_EXPERIMENTS.md](MLService/google_colab/POSITION_CLASSIFIER_EXPERIMENTS.md) |
| Публичная CBCT-когорта (скрипты, данные) | `MLService/tools/`, `MLService/README.md` (раздел про данные); подробные методички — локально вне git |
| API Backend | `Backend/Sources/App/Controllers/`, [Backend/README.md](Backend/README.md) |
| Сеть iOS ↔ Backend | `iOSApp/.../AnalysisEndpoint.swift` (базовый URL и пути) |
| Презентация / экспорт HTML→PDF | локально: каталоги `docs/presentation/`, `scripts/presentation/` (не в git) |

## Репозиторий

- В **git** не попадают личные и сгенерированные файлы (см. `.gitignore`: веса, загрузки, часть локальных каталогов `docs/`, `Nir/`, `scripts/presentation/`, `plan.md` и т.д.).
- Каталог `.cursor/` в репозитории не версионируется; локальные планы Cursor при клонировании могут отсутствовать.
