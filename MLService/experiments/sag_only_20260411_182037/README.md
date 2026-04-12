# sag_only_20260411_182037

Локальная копия артефактов прогона `train_binary_position_classifier.ipynb` (сагитталь, бинарно), распакованная из `sag_only_20260411_182037_bundle.zip` (2026-04-11).

## Сводка (из `training_analysis.json` / `config.json`)

| Поле | Значение |
|------|----------|
| Задача | `sagittal_only_binary` |
| Train / val сэмплов | 142 / 30 |
| Val sag 0 / 1 | 22 / 8 |
| Всего эпох | 46 (early stopping) |
| Best epoch (по val acc) | **6** |
| Best val accuracy | **0.733** |
| Youden J threshold | ~0.506 |
| AUC-ROC (калибровка на val) | **0.472** |
| Accuracy @ Youden threshold | ~0.767 |

**Заметка для разработки:** при 8 non-central в val AUC и порог Youden сильно шумят между прогонами; сравнивать с другими запусками только при том же сплите/seed или на CV.

## Файлы в этой папке

| Файл | Назначение |
|------|------------|
| `best_model.pth` | Веса лучшего чекпоинта |
| `config.json` | Конфиг + калибровка |
| `metrics.jsonl` | Пометрика по эпохам |
| `training_analysis.json` | Полный дамп для анализа |
| `learning_curves.png`, `roc_curve.png`, `training_report.png` | Графики |
| `dataset_preview.png` | Превью кропов (из шага Export) |
| `README_СКАЧАТЬ_ЭКСПЕРИМЕНТ.md` | Описание из ноутбука |

Чекпоинт для инференса/отладки: загрузить `best_model.pth` (ключ `model_state_dict`) в `TMJSagittalClassifier` из ноутбука §5.
