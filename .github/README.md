# GitHub: workflows и настройки

## Workflows

| Файл | Назначение |
|------|------------|
| [workflows/ci.yml](workflows/ci.yml) | CI: фильтр изменённых путей, **ruff** (lint + format check) для MLService, **pytest** для `MLService/tests/`, **`swift build`** для Backend. |
| [workflows/security.yml](workflows/security.yml) | По расписанию / вручную: **pip-audit** зависимостей `MLService/requirements.txt`. |

Триггеры веток смотрите в `on:` внутри каждого workflow.
