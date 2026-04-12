# MLService / training

Общий код обучения: датасеты, лоссы и утилиты, на которые опираются скрипты `train_*.py` в корне `MLService/`.

## Подкаталоги

| Каталог | Содержимое |
|---------|------------|
| `datasets/` | PyTorch `Dataset` для детектора, классификатора положения, heatmap и др. |
| `losses/` | Функции потерь (в т.ч. focal, heatmap MSE). |
| `utils/` | Вспомогательная логика (например, построение heatmap, метрики). |

## Мониторинг обучения

См. [../TRAINING_MONITORING.md](../TRAINING_MONITORING.md).
