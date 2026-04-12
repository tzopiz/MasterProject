# Position Classifier v3 — Balanced + Augmented

**Дата:** 2026-04-05
**Платформа:** Google Colab (CPU fallback, 12 GB RAM)
**Ноутбук:** `MLService/google_colab/train_position_classifier.ipynb`
**Предыдущий:** `position_classifier_v1_baseline/`

## Изменения относительно v1

| Компонент | v1 | v3 |
|---|---|---|
| Loss | CrossEntropyLoss (без весов) | **Weighted** CrossEntropyLoss + label_smoothing=0.1 |
| Backbone | [16, 32, 64, 128] | **[8, 16, 32, 64]** (~4× меньше параметров) |
| Dropout | 0.4 | **0.5** |
| fc_hidden | 128 | **64** |
| Аугментации | np.roll ±3 | **shift + 3D flips + rot90 + gaussian noise + intensity jitter** |

## Конфигурация

| Параметр | Значение |
|---|---|
| Loss weights sagittal | [0.52, 3.89, 1.23] (обратно пропорционально частоте) |
| Loss weights frontal | [1.11, 4.67, 0.53] |
| Label smoothing | 0.1 |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Early stopping | 30 эпох |
| Batch size | 4 |
| Input | (1, 64, 64, 64) |

## Данные

| Сплит | Studies | Crops |
|---|---|---|
| Train | 70 | 140 |
| Val | 16 | 32 |

Распределение классов (train):

| Голова | Class 0 | Class 1 | Class 2 |
|---|---|---|---|
| Sagittal | 90 (64%) | 12 (9%) | 38 (27%) |
| Frontal | 42 (30%) | 10 (7%) | 88 (63%) |

## Результаты

| Метрика | v1 | **v3** |
|---|---|---|
| Лучшая эпоха | 3 | **2** |
| Best val mean accuracy | 0.734 | **0.734** |
| Best val acc sagittal | 0.875 | 0.875 |
| Best val acc frontal | 0.594 | 0.594 |
| Всего эпох | 33 | 32 |
| Train acc (final) | ~0.90 | **~0.55** |
| Train/Val loss gap | ~0.84 | **~0.14** |
| Final LR | 2.5e-05 | 2.5e-05 |

## Анализ

### Что улучшилось

1. **Нет переобучения.** Train accuracy ~0.55 (vs ~0.90 в v1). Модель не запоминает
   тренировочные данные — аугментации и dropout эффективны.

2. **Модель пытается разделять классы.** Sagittal accuracy варьируется от 0.28 до 0.87
   между эпохами вместо фиксированных 0.875 (majority-class collapse в v1).

3. **Train/Val loss gap минимален** (~0.14 vs ~0.84 в v1).

### Что не улучшилось

1. **Потолок accuracy тот же (0.734).** Ни weighted loss, ни аугментации не помогли
   преодолеть лимит генерализации.

2. **Accuracy нестабильна между эпохами.** Sagittal val accuracy: 0.28–0.87. Это
   следствие малого val set (32 кропа) и недостатка данных.

3. **Normal class по-прежнему плохо распознаётся** (4 сэмпла в val).

### Confusion Matrix (epoch 32, НЕ best model — баг исправлен в ноутбуке)

**Sagittal:** модель overcorrected — предсказывает класс 1 (Normal) для 20/28 true class 0.
**Frontal:** класс 0 recall=50%, класс 2 recall=71%, класс 1 recall=0%.

### Корневая причина

**140 кропов** — принципиально недостаточно для 3D CNN даже маленького размера.
Модель не может выучить устойчивые 3D-признаки из такого объёма данных.

## Баг (исправлен)

Confusion matrix и classification report в `training_analysis.json` сняты с модели
**последней эпохи**, а не **лучшей**. Исправлено в ноутбуке: перед анализом
загружается `best_model.pth`.

## Файлы

- `training_analysis.json` — полный отчёт (NB: confusion matrix от epoch 32, не best)
