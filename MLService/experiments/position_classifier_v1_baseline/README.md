# Position Classifier v1 — Baseline (crop-based)

**Дата:** 2026-04-05
**Платформа:** Google Colab (T4 GPU, 12 GB RAM)
**Ноутбук:** `MLService/google_colab/train_position_classifier.ipynb`

## Задача

Классификация положения головки ВНЧС по 3D-кропу вокруг одного сустава.
Каждый кроп — два класса: **sagittal** (3 класса) и **frontal** (3 класса).

## Пайплайн

```
DICOM → TMJDetectorLarge (регрессия центров) → crop 128³ → .npy
  → downsample ×2 → 64³ → TMJCondyleClassifier (2 головы × 3 класса)
```

- Детектор: `experiments/detector_20251126_003305/best_model.pth` (TMJDetectorLarge, MAE=23.9 px)
- Предобработка: slice-streaming (послайсовое чтение, ~20 МБ пик RAM на study)

## Конфигурация

| Параметр | Значение |
|---|---|
| Архитектура | TMJCondyleClassifier: backbone [16,32,64,128] + 2 FC-головы |
| Input | (1, 64, 64, 64) float32 |
| Параметров | ~1М |
| Loss | CrossEntropyLoss (без весов) |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Batch size | 4 |
| Early stopping | 30 эпох без улучшения |
| Аугментация | np.roll ±3 вокселя (только train) |

## Данные

| Сплит | Studies | Crops |
|---|---|---|
| Train | 70 | 140 |
| Val | 16 | 32 |

Сплит по пациентам (нет утечки).

## Результаты

| Метрика | Значение |
|---|---|
| Всего эпох | 33 (early stop) |
| Лучшая эпоха | **3** |
| Best val mean accuracy | **0.734** |
| Best val acc sagittal | 0.875 |
| Best val acc frontal | 0.594 |
| Best val loss | 1.708 |
| Final LR | 2.5e-05 |
| Overfit gap (last 5) | 0.065 |

### Confusion Matrices

**Sagittal** (val, best model):

```
              Predicted
              Ant/Med  Normal  Post/Lat
True Ant/Med    28       0       0
True Normal      4       0       0
True Post/Lat    0       0       0
```

**Frontal** (val, best model):

```
              Predicted
              Ant/Med  Normal  Post/Lat
True Ant/Med     5       0       9
True Normal      4       0       0
True Post/Lat    3       0      11
```

### Classification Report

**Sagittal:**

| Класс | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anterior/Medial | 0.875 | 1.000 | 0.933 | 28 |
| Normal | 0.000 | 0.000 | 0.000 | 4 |
| Posterior/Lateral | — | — | — | 0 |

**Frontal:**

| Класс | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anterior/Medial | 0.417 | 0.357 | 0.385 | 14 |
| Normal | 0.000 | 0.000 | 0.000 | 4 |
| Posterior/Lateral | 0.550 | 0.786 | 0.647 | 14 |

## Анализ и выводы

### Проблемы

1. **Majority-class collapse (sagittal).** Модель предсказывает все 32 val-сэмпла как класс 0.
   Accuracy 0.875 = просто 28/32 (доля majority class). Модель не выучила ничего.

2. **Класс Normal игнорируется обеими головами.** Support: 4 сэмпла в val
   (sagittal) и 4 (frontal). Precision/recall = 0 для обеих.

3. **Переобучение с эпохи 3.** Train loss: 1.90→1.19 (падает), val loss: 1.71→2.03
   (растёт). Gap train_acc - val_acc увеличивается до 0.14.

4. **Мало данных.** 140 train-кропов для 3D CNN ~1М параметров — модель быстро
   запоминает train, не генерализует.

### Рекомендации для v2

1. **Weighted CrossEntropyLoss** — веса обратно пропорциональны частоте класса.
2. **Аугментации** — 3D flips, повороты на 90°, гауссов шум, brightness jitter.
3. **Mirror left↔right** — отзеркалить кропы для удвоения данных.
4. **Меньшая модель** — backbone [8,16,32,64], dropout=0.6.
5. **Label smoothing** — уменьшить уверенность модели в majority class.

## Файлы

- `training_analysis.json` — полный отчёт (history, confusion matrices, predictions)
