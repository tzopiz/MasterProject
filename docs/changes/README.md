# Изменения исследовательского процесса

| Изменение | Статус | Результат |
|---|---|---|
| [003 — TMJ-OD3D](003-tmj-od3d-pilot/README.md) | in-progress | Codebook найден; цель — готовый запуск, полный прогон данных выполняется |
| [004 — Intake](004-od3d-intake/README.md) | done | Реальные CSV/JSON и 58 целевых тестов проверены; итоговый suite 585 passed |
| [005 — Подготовка ROI](005-od3d-preparation/README.md) | in-progress | Ограниченный streaming, приватные crops и resume реализованы; реальная когорта обрабатывается |
| [006 — Обучение](006-od3d-training/README.md) | in-progress | Binary/multilabel CPU-проверки проходят, итоговый preflight ожидает данные |
| [007 — DataSphere](007-osseous-cloud/README.md) | in-progress | Локальный staging и result bindings проверены; финальный bundle ещё не подготовлен |
| [008 — Техническая курация](008-od3d-curation/README.md) | in-progress | Утилита и 25 тестов проверены; реальная review policy ожидает полного источника |

Требования нового пути: [спецификация костных изменений](../../MLService/docs/spec/functional/osseous/README.md).
Незавершённые этапы не считаются принятыми только на основании синтетических тестов.
