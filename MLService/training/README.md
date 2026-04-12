# MLService / training

Общий код обучения: датасеты, лоссы и утилиты, на которые опираются скрипты `train_*.py` в корне `MLService/`.

## Подкаталоги

| Каталог | Содержимое |
|---------|------------|
| `datasets/` | PyTorch `Dataset` для детектора, классификатора положения, heatmap и др. |
| `losses/` | Функции потерь (в т.ч. focal, heatmap MSE). |
| `utils/` | Сиды, бинарные метрики (ROC / Youden), 3D аугментации (`volume_aug_3d.py`), пути DataSphere (`datasphere_env.py`), 2D (`transforms.py`). |
| `sagittal_binary_cv.py` | 5-fold StratifiedGroupKFold CV для сагиттали (бинарно), см. `docs/superpowers/prompts/improve-sag-classifier-metrics.md`. |

## Мониторинг обучения

См. раздел «Мониторинг обучения» в [../README.md](../README.md).
