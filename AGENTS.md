# Сведения для агентов (AI / автоматизация)

Репозиторий **MasterProject** (AI Doctor — анализ ВНЧС по КЛКТ).

## С чего начать

1. **[README.md](README.md)** — продукт, глоссарий, быстрый старт, HTTP API, troubleshooting.
2. **[MLService/README.md](MLService/README.md)** — пайплайн ML, инструменты, карта подпапок.

## Правила документации (обязательно)

- В **каждой каталоге** в git должно быть **либо ровно один** файл **`README.md`**, **либо ни одного** файла с расширением **`.md`**.
- **Исключение для корня репозитория:** допускается второй файл **`AGENTS.md`** (этот файл) — правила для агентов и политика доков.
- Длинные справки, не помещающиеся в один README каталога, выносите в **`.txt`** или в код/комментарии; не плодите второй `.md` рядом с README.
- При расхождении текста с кодом **источник истины — исходники** (`AnalysisController.swift`, `routes.swift`, `app.py`, эндпоинты клиента).

## Границы системы

| Компонент | Путь | Роль |
|-----------|------|------|
| Backend | `Backend/` | Vapor, SQLite, DICOM-серия, вызов ML, результаты |
| ML Service | `MLService/` | FastAPI, PyTorch, детекция / объём |
| iOS | `iOSApp/MasterDoctor/` | SwiftUI-клиент — вход [iOSApp/README.md](iOSApp/README.md) |

Синонимы: **TMJ** = **ВНЧС**, **CBCT** = **КЛКТ**.

## Частые задачи

| Задача | Куда смотреть |
|--------|----------------|
| Запуск цепочки | [README.md](README.md) |
| ML: обучение, датасет, инструменты | [MLService/README.md](MLService/README.md), мониторинг — раздел «Мониторинг обучения» там же |
| Классификатор положения (Colab / DataSphere) | [MLService/google_colab/README.md](MLService/google_colab/README.md); длинная сводка экспериментов — `MLService/google_colab/POSITION_CLASSIFIER_EXPERIMENTS.txt` |
| Метки и модель положения (детально) | [MLService/docs/README.md](MLService/docs/README.md) |
| Публичная когорта | `MLService/tools/`, [MLService/README.md](MLService/README.md) |
| API Backend | `Backend/Sources/App/Controllers/`, [Backend/README.md](Backend/README.md) |
| Сеть iOS ↔ Backend | `iOSApp/.../AnalysisEndpoint.swift` |
| Архитектура iOS (MAA) | `iOSApp/ModularAppArchitecture.txt` |

## Репозиторий

- Тяжёлые артефакты, секреты и часть каталогов — в `.gitignore` / `MLService/.gitignore`.
- Каталог `.cursor/` не версионируется.
