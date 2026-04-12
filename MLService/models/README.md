# Models Directory

Здесь лежат **архитектуры** (`*.py`) и **веса** для инференса.

## Релиз: детектор ВНЧС

Сервис по умолчанию ищет файл **`tmj_detector_best.pth`** в этой папке (см. `app.py`: `MODEL_PATH` или fallback).

- Файл **`tmj_detector_best.pth` не игнорируется** в `.gitignore` — его **нужно закоммитить** перед тегом/релизом, чтобы клон репозитория сразу поднимал ML с реальной моделью (без отдельной загрузки весов).
- Скопируйте свой обученный чекпойнт (например `experiments/detector_*/best_model.pth`) в `models/tmj_detector_best.pth` и добавьте в git:
  ```bash
  cp experiments/detector_YYYYMMDD_HHMMSS/best_model.pth models/tmj_detector_best.pth
  git add models/tmj_detector_best.pth
  ```
- Остальные `*.pth` / `*.pt` / `*.onnx` в `models/` по-прежнему **не** коммитятся (эксперименты, сегментация и т.д.).
- Если вес **> ~100 MiB**, для GitHub используйте **[Git LFS](https://git-lfs.com/)** (`git lfs track "models/tmj_detector_best.pth"` в корне репозитория и закоммитьте `.gitattributes`).

## Поддерживаемые форматы весов

- PyTorch (`.pth`, `.pt`)
- ONNX (`.onnx`)

## Переменная окружения

```bash
export MODEL_PATH=models/tmj_detector_best.pth
```

## Требования к детектору (текущий пайплайн)

Модель загружается через `TMJDetectorService` и `get_detector_model()` — формат чекпойнта см. в `services/detector_service.py` (`model_state_dict` или «сырой» state_dict).

## Режим без весов (dummy)

Если файл по `MODEL_PATH` отсутствует, сервис может работать в упрощённом режиме (см. историю `TMJDetectorService` / логи при старте).

## Обучение и другие веса

Чекпойнты экспериментов остаются под `experiments/` (см. правила в корневом `.gitignore`). Сегментация U-Net и прочие `*_best.pth` в `models/` локально — не для git, кроме **`tmj_detector_best.pth`**.
