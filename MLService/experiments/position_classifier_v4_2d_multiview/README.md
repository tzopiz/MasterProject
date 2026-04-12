# Position Classifier v4 — 2D Multi-View

**Дата:** 2026-04-05
**Платформа:** Google Colab (CPU, 12 GB RAM)
**Ноутбук:** `MLService/google_colab/train_position_classifier_2d.ipynb`
**Предыдущие:** `position_classifier_v1_baseline/`, `position_classifier_v3_balanced/`

## Подход

Вместо 3D CNN на полном кропе — 2D срезы через pretrained ResNet18 (ImageNet).

Из каждого 128-cube `.npy` кропа берутся 3 центральных среза (axial, coronal, sagittal),
каждый resize 224x224, replicate to 3ch, ImageNet normalization → ResNet18 → 512 features.
Итого 3 x 512 = 1536 features на кроп.

Два подхода:
- **Approach C:** Frozen ResNet18 → SVM / Random Forest / Gradient Boosting
- **Approach A:** Fine-tune layer4 + FC heads, weighted loss, 2D augmentations

## Сводка результатов

| Метод | Sagittal | Frontal | Mean | Примечание |
|---|---|---|---|---|
| v1 (3D baseline) | 0.875 | **0.594** | **0.734** | majority-class collapse (sag) |
| v3 (3D balanced) | 0.875 | **0.594** | **0.734** | no overfit, same peak |
| C: SVM (RBF) | 0.688 | **0.594** | 0.641 | frozen features, balanced |
| C: Random Forest | 0.875 | 0.500 | 0.688 | majority-class collapse |
| C: Gradient Boost | 0.625 | 0.438 | 0.531 | — |
| A: Multi-View FT | **0.906** | 0.469 | 0.688 | best sag, but overfit |

## Approach C — Frozen Features + Classical ML

Обучение за секунды, без GPU.

### SVM (RBF) — лучший баланс

- `class_weight='balanced'`, kernel=rbf, C=1.0
- Единственный метод (кроме v1/v3), достигший **frontal = 0.594**
- Sagittal = 0.688 — не коллапсирует в majority class
- Confusion matrix sagittal: 22/28 Ant/Med верно, 6 ошибок → class 2

### Random Forest

- `class_weight='balanced'`, n_estimators=300
- Sagittal = 0.875 → majority-class collapse (all 28 Ant/Med → Ant/Med, all 4 Normal → Ant/Med)
- Frontal = 0.500 — перекос в Post/Lat (recall 0.93, Ant/Med recall 0.21)

### Gradient Boosting

- n_estimators=200, max_depth=4, lr=0.05
- Худший результат: 0.531 mean

## Approach A — Multi-View Fine-Tuning

### Конфигурация

| Параметр | Значение |
|---|---|
| Backbone | ResNet18 (ImageNet), freeze all except layer4 |
| Views | axial + coronal + sagittal (центральные срезы) |
| Heads | 2 x Linear(1536→128→3) + Dropout(0.5) |
| Loss | Weighted CrossEntropyLoss + label_smoothing=0.1 |
| LR | 3e-4, ReduceLROnPlateau (patience=8) |
| Early stopping | 25 эпох |
| Augmentations | h_flip, v_flip, rotation_15, translate_5%, gaussian noise |

### Результаты

- Best epoch: **8** (из 33, early stop)
- Best val acc: **0.688** (sag=0.906, fr=0.469)
- Overfit: train_acc=0.74 vs val_acc=0.20 к эпохе 27

### Confusion Matrix (best model, epoch 8)

**Sagittal** (acc=0.906):

```
              Predicted
              Ant/Med  Normal  Post/Lat
True Ant/Med    28       0       0
True Normal      3       1       0
```

**Frontal** (acc=0.469):

```
              Predicted
              Ant/Med  Normal  Post/Lat
True Ant/Med     5       1       8
True Normal      2       0       2
True Post/Lat    3       1      10
```

### Достижения

1. **Sagittal accuracy 0.906 — рекорд** среди всех подходов (v1/v3 = 0.875)
2. **Первая корректная детекция Normal (sagittal):** 1/4 = 25% recall.
   Все предыдущие модели давали 0% на Normal.
3. Normal precision = 1.0 — когда модель предсказывает Normal, она права.

### Проблемы

1. **Frontal = 0.469 — худший** среди всех подходов
2. **Сильный overfit** — layer4 (~2.5M params) слишком много для 140 samples
3. LR=3e-4 оказался слишком агрессивным

## Выводы

1. **Потолок ~0.734 mean accuracy** сохраняется через все 4 подхода (v1, v3, C, A).
   Это фундаментальное ограничение объёма данных (140 train, 32 val).

2. **2D pretrained features содержат полезную информацию** — frozen ResNet18 + SVM
   достигает frontal = 0.594 без единой эпохи обучения.

3. **Fine-tuning на 140 samples вредит** — Approach A переобучается за 8 эпох.
   Frozen features + SVM практичнее.

4. **Sagittal >> Frontal** по accuracy во всех подходах — anterior/posterior
   положение головки визуально более различимо, чем medial/lateral.

5. **Normal class (~9% train) не поддаётся обучению** при текущем объёме данных.

## Рекомендации

- Для продакшена использовать **SVM на frozen ResNet18 features** — лучший баланс,
  не переобучается, работает без GPU
- При увеличении датасета до ~300 studies — пересмотреть fine-tuning с меньшим LR
- Normal class требует больше данных или объединения с ближайшим классом

## Файлы

- `training_analysis.json` — полный отчёт (Approach C + Approach A)
