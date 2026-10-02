# Исследовательское обучение положения ВНЧС

Общий бинарный/multiclass sagittal-путь использует [tools/run_research.py](../tools/run_research.py) и [train_sagittal_binary_cv.ipynb](train_sagittal_binary_cv.ipynb). CLI и notebook вызывают одинаковые `load_research_config`, `preflight_research`, `run_research`; notebook не содержит второго trainer. Это исследовательский контур, без Backend/iOS интеграции.

## Подготовка входов

Нужны Python 3.12 и зависимости [requirements.txt](../requirements.txt) в отдельном окружении. Проверенный локальный CPU runtime не считается проверкой Linux/CUDA образа DataSphere; каждый прогон записывает фактические версии библиотек. Установка пакетов и получение данных не выполняются notebook автоматически.

Вход — [строгий индекс](../docs/spec/functional/data/README.md) schema version 1 с явно связанными source/study/patient/label-record и подтверждённой применимостью sagittal-меток к исследованию. Нужны обе стороны ROI, созданные совместимыми paired heatmap локализаторами, с проверенными `.passport.json`. Ссылка на архив сама по себе не задаёт patient grouping, применимость меток, ROI происхождение или совместимость детекторов. Crop-only вход подтверждает сохранённые паспорта; исходная DICOM-серия не перепроверяется без `series_path`. Detector training identity и independence остаются unknown.

Подготовленный локальный или read-only mounted каталог указывается явно. Для разрешённой Drive-ссылки официальный [DataSphere guide](https://yandex.cloud/en/docs/datasphere/operations/data/connect-to-google-drive) описывает `gdown`. Пример ручного получения в приватный staging-каталог, при установленном `gdown`:

```bash
export DATASET_URL='<authorized private URL>'
gdown --folder "$DATASET_URL" --output ./data/staged
```

Доступ, квота и сеть проверяются для переданной ссылки; mount или локальный staging остаётся возможным источником. Реальная ссылка не хранится в Git. Получение raw DICOM не делает набор готовым к обучению: сначала строгий индекс и проверенные ROI. Автоматический перенос и подтверждённый платный Job принадлежат отдельному cloud-launch пути.

## Один приватный config

Сохраните полный конфиг ниже как `research.private.json` рядом с каталогом `inputs/`; это синтетические имена. `inputs/index.private.json` содержит строгий индекс, `inputs/` — его файлы, `runs/` — отдельный writable output. Все относительные пути разрешаются от config file, независимо от cwd. Входной каталог и output не могут содержать друг друга; output также не может содержать input index.

```json
{
  "schema_version": 1,
  "mode": "binary",
  "input_path": "inputs/index.private.json",
  "dataset_root": "inputs",
  "output_dir": "runs",
  "run_id": null,
  "n_splits": 5,
  "seed": 42,
  "epochs": 80,
  "batch_size": 16,
  "lr": 0.00003,
  "weight_decay": 0.0001,
  "early_stopping_patience": 25,
  "lr_plateau_patience": 5,
  "lr_plateau_factor": 0.5,
  "max_grad_norm": 1.0,
  "num_workers": 0,
  "gamma": 2.0,
  "features": [8, 16, 32, 64],
  "fc_hidden": 128,
  "dropout": 0.5,
  "train_augment_mode": "strong",
  "device": null,
  "tqdm_disable": false,
  "log_each_epoch": true,
  "log_epochs_jsonl": true
}
```

Обязательны schema_version/mode и три пути; остальные поля получают показанные defaults из `SagittalBinaryCVConfig`, resolved значения записываются в private replay. `device=null` выбирает CUDA, затем MPS, затем CPU; `"cpu"` задаёт CPU явно. Preflight проверяет синтаксис device, но не доступность GPU. ROI размер и preprocessing определяются проверенным паспортом и не меняются конфигом; runner не выполняет resampling. `train_augment_mode` принимает none/flip_only/strong для binary. Для multiclass задайте `"mode":"multiclass"` и `"train_augment_mode":"none"` в том же конфиге; если augmentation field отсутствует, multiclass default none. Binary default strong сохраняется. Explicit flip_only/strong в multiclass отказывают; оси anterior/posterior не ремапятся догадкой. Multiclass focal gamma не используется: loss — обычный cross-entropy.

Unknown keys, duplicate JSON, неверные типы, NaN/Infinity и неизвестная версия/режим блокируют запуск. Старые manifest/name-join настройки не входят в этот config. Неверные явно заданные TMJ_DATASET_DIR/TMJ_CROP_DIR/TMJ_MANIFEST_PATH/TMJ_LABELS_PATH/ML_SERVICE_ROOT блокируют запуск безопасным кодом. Валидные legacy TMJ_* не заменяют canonical config. Auto-discovery доступно только legacy helper при unset значении; несколько кандидатов дают ошибку, отсутствующий явный путь не заменяется другим набором.

## Проверка и запуск

Из `MLService/`, в подготовленном окружении:

```bash
python -m tools.run_research --config /private/staging/research.private.json --preflight
python -m tools.run_research --config /private/staging/research.private.json
```

`--preflight` не создаёт модель, output или probe files. Он проверяет индекс, paired ROI и пригодность patient folds, печатает только counts/status; обе стороны и повторные исследования пациента остаются в одной группе. CLI печатает машинно-читаемый JSON; training progress идёт в stderr. Ошибка содержит фиксированный код, без входных ID/путей. В notebook запускайте kernel из `MLService` либо настройте PYTHONPATH на этот каталог; допустим явный ML_SERVICE_ROOT. Затем укажите CONFIG_PATH и выполните preflight. Последняя ячейка запускает тот же callable и печатает только агрегаты.

Read-only inputs поддерживаются; ближайший существующий ancestor output проверяется writable без записи. Фактический run каталог создаёт CV с защитой от повторного использования. `run_id=null` генерирует новый opaque ID; повтор заданного run_id отказывает. Нет resume, поэтому после отказа корректируют вход/config и начинают новый run.

## Артефакты и границы оценки

Каждый каталог прогона содержит atomic `report.json`, `epochs.jsonl`, fold checkpoints и приватные `replay.private.json` / `fold_XX_predictions.private.jsonl`. Публичный report содержит агрегаты и digests, без raw IDs/paths или patient rows. Checkpoints остаются локальными, без auto-publication. Replay, membership, per-crop hashes и predictions не публикуются даже после псевдонимизации; хеши не означают полную анонимность. Не копируйте весь run каталог в Git или публичный отчёт.

Binary выбирает checkpoint по validation AUC, отдельный неаугментированный training loader используется для Youden threshold calibration. Multiclass выбирает checkpoint по validation macro F1; scores within 1e-8 сохраняют раннюю эпоху. Решение multiclass — argmax softmax, ties lowest class index, без Youden/калибровки. Поэтому `development_cv` не является независимой test-оценкой. Partial/failed run сохраняет завершённые folds и безопасный status. Метрики можно восстановить из private predictions; CPU reload и контракт checkpoint описаны в [models](../docs/ai/models/README.md), анализ агрегатного report — в [evaluation](../docs/ai/evaluation/README.md).

Проверены strict config/path отказы, model-free preflight, единая семантика CLI/notebook и настоящие tiny binary/multiclass CPU CLI runs на synthetic NIfTI с паспортами. Такие проверки подтверждают технический путь, не качество медицинской классификации. Live GPU/cloud, доступность реальной ссылки, совместимость локализаторов, пригодность реальной когорты и критерий качества проверяются отдельно.

## DataSphere

[Jobs](https://yandex.cloud/en/docs/datasphere/operations/projects/work-with-jobs) выполняются на отдельной Linux x86_64 VM, независимо от JupyterLab. Подготовьте Python 3.12 окружение и объявленные private inputs/outputs; notebook mount не подразумевает наличие тех же путей в Job. [Project disk](https://yandex.cloud/en/docs/datasphere/concepts/jobs/) монтируется read-only через DS_PROJECT_HOME. Новый dataset resource не нужен; создание таких ресурсов прекращено после 2026-04-20. Config использует фактический mount/staging path и отдельный writable output. Здесь нет CLI submission, cloud SDK, credentials, hardcoded project ID или автоматического расходования ресурсов.

## Исторические notebook

| Файл | Назначение и граница |
|---|---|
| [train_position_classifier.ipynb](train_position_classifier.ipynb) | Старый 3D whole-volume путь; не canonical per-side ROI runner. |
| [train_position_classifier_2d.ipynb](train_position_classifier_2d.ipynb) | 2D multi-view/feature experiments; отдельный исследовательский протокол. |
| [train_binary_position_classifier.ipynb](train_binary_position_classifier.ipynb) | Самостоятельный исторический binary experiment; не единый config owner. |
| [init_datasphere_dataset.ipynb](init_datasphere_dataset.ipynb) | Историческая инициализация; не инструкция создавать новый dataset resource. |

Исторические результаты и ограничения: [POSITION_CLASSIFIER_EXPERIMENTS.txt](POSITION_CLASSIFIER_EXPERIMENTS.txt), [experiments](../experiments/README.md). Их успешность не доказывает готовность или качество нового run.


## Классы общего ROI multiclass

Исходные sagittal codes 1/2/3 отображаются strict intake в 0/1/2:
central/anterior/posterior. Каждая сторона — отдельный sample, весь пациент —
одна source-qualified group. Stratification использует максимальный class index
пациента как heuristic; preflight дополнительно требует поддержку всех трёх
классов в каждом train/validation fold и не пробует другой split молча. Если
данных недостаточно, уменьшение folds или состав когорты решают до запуска.

Frontal codes 4/5/6 остаются optional отдельной разметкой; текущий multiclass
обучает только sagittal. Модель возвращает (sagittal logits[B,3],None), без
frontal head и неподготовленных frontal predictions. Missing frontal не создаёт
normal label. Legacy whole-volume four-head trainer остаётся историческим путём.

Fold report включает allclass support, per-class precision/recall/F1, true-row /
predicted-column confusion matrix, macro F1 и balanced accuracy. Majority baseline
выбирает класс только по training counts, ties lowest class index. Summary —
mean/population std по fold metrics, не confidence interval. Private predictions
хранят aligned3-class logits/probabilities/argmax decisions; public report только
агрегаты. Shared reload проверяет mode/family/classes/tasks/preprocessing и weights;
такой synthetic reload подтверждает техническую совместимость, не clinical quality.
