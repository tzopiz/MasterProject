# Документация проекта AI Doctor (MasterProject)

Центральная точка входа: навигация по материалам для разработчиков и для ИИ-агентов.

## Быстрый старт

| Документ | Назначение |
|----------|------------|
| [../README.md](../README.md) | Обзор системы, запуск Backend и ML Service, конфигурация |
| [../QUICKSTART.md](../QUICKSTART.md) | Пошаговый запуск ML → Backend → iOS, текущие метрики детектора |
| [../TMJ_DETECTION_SETUP.md](../TMJ_DETECTION_SETUP.md) | Настройка конвейера детекции TMJ, endpoints, troubleshooting |

## Данные: публичная CBCT-когорта (метки классов ВНЧС)

| Документ | Назначение |
|----------|------------|
| [cbct-public-cohort-dataset.md](cbct-public-cohort-dataset.md) | Пайплайн DOCX → JSON меток → Яндекс.Диск → zip → распаковка → очистка → организация датасета и анонимизация DICOM; скрипты, пути, манифесты |

## Контекст для агентов и новых участников

| Документ | Назначение |
|----------|------------|
| [../AGENTS.md](../AGENTS.md) | Краткая памятка: что читать первым, таблица компонентов |
| [project-context.md](project-context.md) | Глоссарий, схема потока данных, карта репозитория, HTTP API (с оговорками) |

## Компоненты (детально)

| Область | Файл |
|---------|------|
| Backend (Vapor) | [../Backend/README.md](../Backend/README.md) |
| ML Service | [../MLService/README.md](../MLService/README.md) |
| Обучение и мониторинг | [../MLService/TRAINING_MONITORING.md](../MLService/TRAINING_MONITORING.md) |
| Валидация | [../MLService/validation/README.md](../MLService/validation/README.md) |
| Эксперименты | [../MLService/experiments/README.md](../MLService/experiments/README.md) |
| Модели (чекпойнты) | [../MLService/models/README.md](../MLService/models/README.md) |
| Инструмент ROI | [../MLService/tools/README_ROI_TOOL.md](../MLService/tools/README_ROI_TOOL.md) |
| Классификация TMJ (tool) | [../MLService/tools/tmj_classification_tool/README.md](../MLService/tools/tmj_classification_tool/README.md) |

## iOS и архитектура приложения

| Документ | Назначение |
|----------|------------|
| [../iOSApp/ModularAppArchitecture.md](../iOSApp/ModularAppArchitecture.md) | Модульная структура Xcode-проекта |

## Презентации и материалы конференции

| Путь | Назначение |
|------|------------|
| [presentation/PROJECT_PRESENTATION.md](presentation/PROJECT_PRESENTATION.md) | Краткое описание проекта для защиты / презентаций |
| [presentation/EXPORT_README.md](presentation/EXPORT_README.md) | Экспорт `presentation.html` в PDF/PPTX |
| [presentation/presentation.html](presentation/presentation.html), [presentation/presentation_static.html](presentation/presentation_static.html) | Исходники слайдов |
| [presentation/доклад_конференция_начальный_этап.md](presentation/доклад_конференция_начальный_этап.md), [presentation/план_доклада_конференция.md](presentation/план_доклада_конференция.md), [presentation/тезисы_конференция.txt](presentation/тезисы_конференция.txt) | Тексты для конференции |
| [../scripts/presentation/](../scripts/presentation/) | Скрипты: DOCX/PPTX, экспорт HTML→PDF/PPTX (`export_requirements.txt`) |

## Прочее

| Путь | Содержание |
|------|------------|
| `../Nir/` | Материалы по НИР (в т.ч. отчёты в Markdown) |
| `../examples/` | Примеры данных или сценариев (смотреть содержимое каталога) |
