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
