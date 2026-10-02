# GitHub: workflows и настройки

## Workflows

| Файл | Назначение |
|------|------------|
| [workflows/specs.yml](workflows/specs.yml) | Спецификации: Python stdlib + Git, self-test, структура, файловые ссылки/якоря, уникальность ID и связи с приёмкой. Выполняется независимо от путей компонентов; модель не запускается. |
| [workflows/ci.yml](workflows/ci.yml) | CI: фильтр изменённых путей, **ruff** (lint + format check) для MLService, **pytest** для `MLService/tests/` на Python 3.10–3.12, отдельный Python 3.12 CPU smoke исследовательского пути с закреплёнными зависимостями, **`swift build`** для Backend. Синтетические проверки не передают данные в облако. |
| [workflows/security.yml](workflows/security.yml) | По расписанию / вручную: **pip-audit** зависимостей `MLService/requirements.txt`. |

Триггеры веток смотрите в `on:` внутри каждого workflow.

Локально: `python3 scripts/check_specs.py --self-test` и `python3 scripts/check_specs.py`. [Область проверки](../scripts/README.md). Эта проверка не подтверждает выполнение поведенческих тестов или достижение метрик ML.
