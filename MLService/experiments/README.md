# Experiments Directory

Эта директория содержит результаты всех экспериментов по обучению моделей.

## Структура эксперимента

Каждый эксперимент должен быть в отдельной папке с именем `exp_XXX_description/`:

```
experiments/
├── exp_001_baseline/
│   ├── config.yaml           # Конфигурация эксперимента
│   ├── train_log.txt         # Лог обучения
│   ├── metrics.json          # Финальные метрики
│   ├── plots/                # Графики (loss, metrics)
│   │   ├── train_loss.png
│   │   ├── val_loss.png
│   │   └── dice_score.png
│   ├── checkpoints/          # Сохраненные веса
│   │   ├── model_best.pth
│   │   ├── model_epoch_10.pth
│   │   └── model_epoch_20.pth
│   └── samples/              # Примеры предсказаний
│       ├── sample_001.png
│       └── sample_002.png
└── exp_002_with_augmentation/
    └── ...
```

## Naming Convention

- `exp_001_baseline` - Baseline модель без аугментации
- `exp_002_augmented` - С аугментацией
- `exp_003_attention_unet` - Архитектура с attention
- `exp_004_3d_unet` - 3D версия модели
- и т.д.

## Шаблон config.yaml

```yaml
experiment:
  name: "exp_001_baseline"
  description: "Baseline U-Net training on processed_crops dataset"
  date: "2025-11-23"

model:
  architecture: "UNet"
  in_channels: 1
  out_channels: 1
  
data:
  train_dir: "data/processed_crops"
  val_split: 0.2
  test_split: 0.1
  
training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.0001
  optimizer: "Adam"
  loss: "BCE + Dice"
  early_stopping: true
  patience: 10
  
augmentation:
  enabled: false
  rotation: 15
  flip: true
  elastic: false
  
results:
  best_epoch: 45
  best_val_dice: 0.87
  final_train_loss: 0.12
  final_val_loss: 0.15
```

## Как создать новый эксперимент

1. Создайте папку с именем `exp_XXX_description/`
2. Скопируйте шаблон `config.yaml`
3. Запустите обучение с указанием output директории
4. Все логи, веса и визуализации сохранятся автоматически

## Tracking экспериментов

Рекомендуется использовать:
- **TensorBoard** для визуализации в реальном времени
- **Weights & Biases** для cloud tracking
- **MLflow** для версионирования моделей

## Best Practices

1. **Всегда заполняйте config.yaml** перед обучением
2. **Сохраняйте промежуточные checkpoints** каждые N эпох
3. **Визуализируйте результаты** после обучения
4. **Документируйте выводы** в README внутри папки эксперимента
5. **Не удаляйте старые эксперименты** - они нужны для сравнения

## Метрики для оценки

### Segmentation Model
- **Dice Coefficient** (main metric)
- **IoU** (Intersection over Union)
- **Precision** (точность)
- **Recall** (полнота)
- **Hausdorff Distance** (граничные ошибки)

### Training Metrics
- Train/Val Loss
- Learning Rate (если используется scheduler)
- Time per epoch
- GPU/CPU usage

---

**Note**: Эта папка добавлена в `.gitignore` (кроме README.md) для экономии места в репозитории. Используйте Git LFS или облачное хранилище для версионирования больших файлов моделей.

