# Models Directory

Здесь лежат **архитектуры** (`*.py`) и **веса** для инференса.

## Релизный детектор ВНЧС

`app.py` по умолчанию ожидает **`tmj_detector_best.pth`** в этой папке (или `MODEL_PATH` / последний `experiments/detector_*/best_model.pth`).

- Веса в **`MLService/models/`** можно хранить в git вместе с кодом (ограничения размера репозитория GitHub — см. [документацию](https://docs.github.com/en/repositories/working-with-files/managing-large-files); при необходимости [Git LFS](https://git-lfs.com/)).
- В корне репозитория каталог **`models/`** (не внутри MLService) по-прежнему в корневом `.gitignore` для локальных больших файлов.

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

Дополнительные чекпойнты (U-Net и т.д.) можно держать в этой папке и коммитить; тяжёлые прогоны по-прежнему удобнее оставлять в `experiments/` (см. правила в корневом `.gitignore` для `MLService/experiments/`).
