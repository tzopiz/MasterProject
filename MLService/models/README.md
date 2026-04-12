# Models Directory

Здесь лежат **архитектуры** (`*.py`) и **веса** для инференса.

## Релизный детектор ВНЧС

`app.py` по умолчанию ожидает **`tmj_detector_best.pth`** в этой папке (или `MODEL_PATH` / последний `experiments/detector_*/best_model.pth`).

- Файл **`tmj_detector_best.pth`** **разрешён в git** (исключение в `.gitignore`) — его можно закоммитить для релиза «из коробки».
- Остальные `*.pth` / `*.pt` / `*.onnx` в `models/` по-прежнему не коммитятся.
- Крупные веса на GitHub — при необходимости [Git LFS](https://git-lfs.com/).

```bash
cp experiments/<прогон>/best_model.pth models/tmj_detector_best.pth
git add models/tmj_detector_best.pth
```

## Форматы

- PyTorch (`.pth`, `.pt`), ONNX (`.onnx`)

## Переменная окружения

```bash
export MODEL_PATH=models/tmj_detector_best.pth
```

Загрузка чекпойнта — см. `services/detector_service.py` (`model_state_dict` или «сырой» state dict).

## Режим без весов

Если файл по пути не найден, детектор может не подняться — смотрите логи при старте `app.py`.

## Сегментация и прочие веса

U-Net и экспериментальные чекпойнты — в `experiments/` и локально в `models/`; в git не тащите, кроме **`tmj_detector_best.pth`**.
