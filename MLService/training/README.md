# MLService / training

Общий код обучения: датасеты, лоссы и утилиты, на которые опираются скрипты `train_*.py` в корне `MLService/`.

## Подкаталоги

| Каталог | Содержимое |
|---------|------------|
| `datasets/` | PyTorch `Dataset` для детектора, классификатора положения, heatmap и др. |
| `losses/` | Функции потерь (в т.ч. focal, heatmap MSE). |
| `utils/` | Сиды, бинарные метрики (ROC / Youden), 3D аугментации (`volume_aug_3d.py`), пути DataSphere (`datasphere_env.py`), 2D (`transforms.py`). |
| `sagittal_binary_cv.py` | 5-fold StratifiedGroupKFold CV для сагиттали (бинарно), см. `docs/superpowers/prompts/improve-sag-classifier-metrics.md`. |

### Выходные файлы CV (`output_json` + анализ)

Пусть `output_json` = `…/experiments/sagittal_cv_last.json` (типичный путь через `default_cv_output_json` на DataSphere).

- **`sagittal_cv_last.json`** — итог и снимки по фолдам: `folds`, `epoch_history` внутри каждого фолда, `summary`, поля прогресса при пофолдовой записи.
- **`sagittal_cv_last_epochs.jsonl`** — при `log_epochs_jsonl=True`: append **по каждой эпохе** (метрики + `fold`).
- **Разбор** — функция `analyze_sagittal_cv_result` (ноутбук `google_colab/train_sagittal_binary_cv.ipynb`, нижняя ячейка): при `report_path=…/sagittal_cv_last_analyze` рядом появляются `*_analyze.txt`, `*_analyze_export.json`, `*_analyze_folds.csv`, `*_analyze_epochs.csv`, `*_analyze_curves.png`.

Подробная таблица имён — в [../google_colab/README.md](../google_colab/README.md) (раздел «Артефакты CV сагиттали»).

## Мониторинг обучения

См. раздел «Мониторинг обучения» в [../README.md](../README.md).

Для дальнейших osseous экспериментов используйте `train(config, development=True)`
или `python -m tools.run_osseous_research --config <private.json> --develop`
из MLService. Нужен `split_path` с уже закреплённым patient split. Этот режим
не открывает test crops и явно обозначает validation результаты как development.

## Frozen osseous features

[Offline API](tmj_osseous_features.py): `load_backbone(local_weights)`,
`extract_features(backbone, crop)`, `fit_head(X, y, mode="binary", C=.01)`
и `predict_head(bundle, X)`. Для multilabel передайте `mode="multilabel"`
и targets[N,6] в порядке авторских кодов1–6. Caller передаёт только fit-строки
при обучении scaler/head и хранит bundle приватно. Public CLI/CV orchestration
для этого пути ещё не реализованы; scratch development CLI выше имеет другой
контракт. [Форматы и приёмка](../docs/spec/functional/osseous/README.md#fr-ml-osseous-features).

Для extractor нужны совместимые PyTorch/torchvision и локальные официальные
ResNet18 ImageNet1K V1 weights с указанным в контракте полным checksum;
автоматической установки/download нет. Synthetic suite не требует этих весов.
