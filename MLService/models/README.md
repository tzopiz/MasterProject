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


## Исследовательский binary sagittal checkpoint

[TMJBinaryPositionClassifier](tmj_binary_position_classifier.py) сохраняется
действующей [CV](../training/sagittal_binary_cv.py) в отдельном локальном run
каталоге. Контракт schema v1 содержит family, точные architecture args,
`trained_tasks=["sagittal"]`, class 0=central / 1=non-central, preprocessing
`tmj-binary-nifti-percentile-v1`, selected epoch, CPU state и точный Youden
threshold. Решение: `sigmoid(sag_logit) >= threshold`; frontal output не обучен.

`load_binary_position_checkpoint(path, device="cpu", expected_sha256=None)`
возвращает `(model.eval(), metadata)`, реконструируя параметры сохранённой
архитектуры. Передайте SHA-256 checkpoint из report для проверки целостности.
Загрузчик использует `weights_only=True` и strict state shape validation;
несовместимые family/config/preprocessing, нечисловой threshold, повреждённый
checkpoint или несовпадение digest дают безопасный `invalid_binary_checkpoint`.
Legacy «сырой» state dict этим loader не принимается.

В checkpoint metadata нет patient IDs, входных paths или private config.
Веса и все outputs остаются локальными; публикация отдельно разрешается
владельцем исследования. Membership и per-case predictions остаются приватными
даже с hashed keys. Контракт не подключает классификатор к HTTP/app и не
подтверждает независимость от detector training или медицинское качество.
Подробности хранения и пересчёта: [оценка](../docs/ai/evaluation/README.md).

Общий [research architecture guard](blocks.py) применяется до constructor в classifier и paired detector loaders, а также в training preflight/CV: один входной канал, 1..4 encoder stages, 1..256 каналов на stage; classifier hidden head 1..2048. Detector bottleneck может удвоить последний stage до512. Для каждого MaxPool3d(2) crop edge должен быть ≥2**число classifier stages; canonical CV/preflight и portable inference проверяют это по validated ROI contract до model/output. Это сохраняет текущие defaults и tiny fixtures и отклоняет unbounded metadata args до allocation; strict state loading и weights_only остаются обязательными. Ограничение параметров не является обещанием памяти для training activations. Регрессии: [allocation tripwires](../tests/test_research_architecture.py).
