# Validation Directory

Инструменты для валидации и оценки ML моделей.

## Структура

```
validation/
├── README.md              # Этот файл
├── metrics.py             # Метрики для оценки (Dice, IoU, etc.)
├── visualize.py           # Визуализация результатов
├── evaluate_model.py      # Комплексная оценка модели
├── baseline.py            # Baseline методы для сравнения
└── reports/               # Сгенерированные отчеты
    ├── exp_001_report.html
    └── comparison.csv
```

## Компоненты

### 1. metrics.py
Все метрики для оценки качества сегментации:
- **Dice Coefficient** - основная метрика overlap
- **IoU (Jaccard Index)** - intersection over union
- **Precision** - точность предсказаний
- **Recall** - полнота детекции
- **Specificity** - true negative rate
- **Hausdorff Distance** - максимальное расстояние между границами
- **Average Surface Distance** - среднее расстояние между поверхностями

### 2. visualize.py
Функции для визуализации:
- Overlay маски на изображение
- Side-by-side сравнение (ground truth vs prediction)
- 3D rendering сегментаций
- Heatmap ошибок
- ROC curves
- Confusion matrices (для multi-class)

### 3. evaluate_model.py
Комплексная оценка модели:
- Загрузка модели и test dataset
- Вычисление всех метрик
- Генерация визуализаций
- Создание HTML отчета
- Сохранение результатов в CSV/JSON

### 4. baseline.py
Baseline методы для сравнения:
- Simple thresholding
- Otsu's method
- Watershed segmentation
- Traditional CV approaches

## Использование

### Быстрая оценка модели
```bash
cd MLService
python validation/evaluate_model.py \
    --model models/segmentation_model_best.pth \
    --data data/test_crops \
    --output validation/reports/exp_001_report.html
```

### Сравнение нескольких моделей
```bash
python validation/compare_models.py \
    --models models/model_v1.pth models/model_v2.pth \
    --data data/test_crops \
    --output validation/reports/comparison.csv
```

### В коде Python
```python
from validation.metrics import dice_coefficient, iou_score
from validation.visualize import plot_segmentation_overlay

# Вычислить метрики
dice = dice_coefficient(pred_mask, gt_mask)
iou = iou_score(pred_mask, gt_mask)

# Визуализировать
plot_segmentation_overlay(image, pred_mask, gt_mask, 
                         title=f"Dice: {dice:.3f}")
```

## Метрики - Детали

### Dice Coefficient
```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```
- Диапазон: [0, 1], где 1 = perfect overlap
- Целевое значение: > 0.85 для медицинской сегментации

### IoU (Jaccard Index)
```
IoU = |A ∩ B| / |A ∪ B|
```
- Диапазон: [0, 1]
- Целевое значение: > 0.75

### Hausdorff Distance
```
HD = max(h(A, B), h(B, A))
```
где h(A, B) = max(min(d(a, b)))
- Измеряет максимальную ошибку на границах
- Единица: пиксели или мм (с учетом spacing)
- Чем меньше, тем лучше

## Test Dataset

Для валидации нужен отдельный test set, который **никогда не использовался** при обучении:

```
data/test_crops/
├── test_001_left.nii.gz
├── test_001_left_mask.nii.gz
├── test_001_right.nii.gz
├── test_001_right_mask.nii.gz
└── ...
```

**Рекомендации**:
- Выделить 10-20% от всех данных
- Сбалансировать по сложности (легкие и сложные случаи)
- Не смотреть на test set до финального тестирования

## Baseline для сравнения

Важно сравнить ML модель с простыми методами:

1. **Random Baseline**: Случайная маска
2. **Simple Threshold**: Пороговая сегментация (Otsu)
3. **Region Growing**: Рост областей
4. **Watershed**: Алгоритм водораздела

Если ML модель не превосходит baseline - нужно улучшать модель или данные.

## Генерация отчетов

Отчет должен содержать:
1. **Метрики**: Таблица со всеми метриками
2. **Визуализации**: Примеры предсказаний (лучшие и худшие)
3. **Статистика**: Распределение метрик, outliers
4. **Ошибки**: Типичные ошибки модели
5. **Сравнение**: С baseline и предыдущими версиями

---

**Note**: Результаты валидации критически важны для магистерской работы. Документируйте все эксперименты!

