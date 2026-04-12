# MLService / tests

Юнит- и интеграционные тесты на **pytest**.

## Запуск

Из каталога **`MLService/`** с активированным venv:

```bash
python -m pytest tests/ -v --tb=short
```

Отдельный файл:

```bash
python -m pytest tests/test_focal_loss.py -v
```

Те же команды выполняются в CI (см. `.github/workflows/ci.yml`).
