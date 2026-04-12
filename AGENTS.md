# Сведения для агентов (AI / автоматизация)

Краткий вход в репозиторий **MasterProject** (AI Doctor — анализ ВНЧС по КЛКТ).

## С чего начать

1. **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** — термины, поток данных, карта каталогов, HTTP-маршруты (при сомнениях сверяйтесь с кодом).
2. **[README.md](README.md)** — обзор и **быстрый старт** (ML → Backend → iOS).

## Границы системы

| Компонент | Путь | Роль |
|-----------|------|------|
| Backend | `Backend/` | Vapor, SQLite, приём DICOM-серии, вызов ML, выдача результатов |
| ML Service | `MLService/` | FastAPI, PyTorch, детекция/обработка объёма |
| iOS | `iOSApp/MasterDoctor/` | SwiftUI-клиент |

Доменные синонимы: **TMJ** = **ВНЧС**, **CBCT** = **КЛКТ**.

## Правило актуальности

При расхождении текста с кодом **источник истины — исходники** (`AnalysisController.swift`, `routes.swift`, `app.py`, эндпоинты в клиенте). Нюансы — в [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## Частые задачи

| Задача | Куда смотреть |
|--------|----------------|
| Запуск всей цепочки | [README.md](README.md) (раздел «Быстрый старт»), [TMJ_DETECTION_SETUP.md](TMJ_DETECTION_SETUP.md) |
| ML: обучение, датасет, инструменты | [MLService/README.md](MLService/README.md), [MLService/TRAINING_MONITORING.md](MLService/TRAINING_MONITORING.md) |
| Классификатор положения ВНЧС (Colab / DataSphere) | [MLService/google_colab/README.md](MLService/google_colab/README.md), [MLService/google_colab/POSITION_CLASSIFIER_EXPERIMENTS.md](MLService/google_colab/POSITION_CLASSIFIER_EXPERIMENTS.md) |
| Публичная CBCT-когорта (скрипты, данные) | `MLService/tools/`, `MLService/README.md` |
| API Backend | `Backend/Sources/App/Controllers/`, [Backend/README.md](Backend/README.md) |
| Сеть iOS ↔ Backend | `iOSApp/.../AnalysisEndpoint.swift` |

## Репозиторий

- В **git** не попадают тяжёлые артефакты, секреты и часть локальных каталогов (см. `.gitignore`).
- Каталог `.cursor/` не версионируется.
