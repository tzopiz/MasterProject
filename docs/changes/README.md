# Изменения исследовательского процесса

| Изменение | Статус | Результат |
|---|---|---|
| [016 — Max pooling](016-osseous-max-pooling/README.md) | done | CPU experiment завершён, max16 отклонён; reviewed результаты опубликованы в PR115 |
| [015 — Layer4 fine-tuning](015-osseous-layer4/README.md) | done | Layer4 library,31 CPU checks/private replay; PR113 опубликован, GPU CV отдельно |
| [014 — A100 staging](014-osseous-a100/README.md) | done | Выбор g2.1, resource-specific quote/date;42 offline checks |
| [013 — Frozen features](013-osseous-frozen-features/README.md) | done | Offline feature/head API,31 synthetic checks и точный private replay; CLI отдельно |
| [012 — Архитектура](012-osseous-architecture/README.md) | done | Явный experimental spatial_head; default и legacy provenance сохранены |
| [011 — Development](011-osseous-development/README.md) | done | Train/validation изоляция, stable loss, 51 checks; TRAIN-only диагностика отдельно |
| [003 — TMJ-OD3D](003-tmj-od3d-pilot/README.md) | done | Готовый запуск выбранной когорты; добор остановлен пользователем |
| [004 — Intake](004-od3d-intake/README.md) | done | Реальные CSV/JSON и 58 целевых тестов проверены; итоговый suite 585 passed |
| [005 — Подготовка ROI](005-od3d-preparation/README.md) | done | 148 source cases обработаны; полная загрузка не заявляется |
| [006 — Обучение](006-od3d-training/README.md) | done | Оба real preflight, frozen split и staged CPU step пройдены |
| [007 — DataSphere](007-osseous-cloud/README.md) | done | Два реальных immutable пакета готовы; платный запуск ожидает подтверждения |
| [008 — Техническая курация](008-od3d-curation/README.md) | done | Два source failures явно исключены; policy/provenance сохранены |
| [010 — CLI requirements](010-datasphere-requirements/README.md) | done | Isolated imports/pins/CUDA подтверждены реальным job; training outcome отдельно |
| [009 — Выбранная когорта](009-bounded-cohort/README.md) | done | 146 пациентов / 289 сторон, оба режима готовы к запуску |

Требования нового пути: [спецификация костных изменений](../../MLService/docs/spec/functional/osseous/README.md).
Незавершённые этапы не считаются принятыми только на основании синтетических тестов.
