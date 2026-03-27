# Документация проекта AI Doctor (MasterProject)

Центральная точка входа: навигация по материалам для разработчиков и для ИИ-агентов.

## Быстрый старт

| Документ | Назначение |
|----------|------------|
| [../README.md](../README.md) | Обзор системы, запуск Backend и ML Service, конфигурация |
| [../QUICKSTART.md](../QUICKSTART.md) | Пошаговый запуск ML → Backend → iOS, текущие метрики детектора |
| [../TMJ_DETECTION_SETUP.md](../TMJ_DETECTION_SETUP.md) | Настройка конвейера детекции TMJ, endpoints, troubleshooting |

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

## Презентации и текстовые материалы (корень репозитория)

| Файл | Назначение |
|------|------------|
| [../PROJECT_PRESENTATION.md](../PROJECT_PRESENTATION.md) | Краткое описание проекта для защиты / презентаций |
| [../EXPORT_README.md](../EXPORT_README.md) | Экспорт `presentation.html` в PDF/PPTX |
| `presentation.html`, `presentation_static.html` | Исходники слайдов |
| `доклад_конференция_начальный_этап.md`, `план_доклада_конференция.md`, `тезисы_конференция.txt` | Тексты для конференции |

## Прочее

| Путь | Содержание |
|------|------------|
| `../Nir/` | Материалы по НИР (в т.ч. отчёты в Markdown) |
| `../examples/` | Примеры данных или сценариев (смотреть содержимое каталога) |
