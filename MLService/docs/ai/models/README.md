# Модели: несовместимые пути

| Семейство | Проверенный путь | Выход / извлечение |
|---|---|---|
| Регрессия | [tmj_detector](../../../models/tmj_detector.py), [detector service](../../../services/detector_service.py) | 6 sigmoid координат left zyx, right zyx; HTTP масштабирует по original shape |
| Общая heatmap | [tmj_heatmap_detector](../../../models/tmj_heatmap_detector.py), [trainer](../../../train_heatmap_detector.py) | По умолчанию 2 канала logits; [utilities](../../../training/utils/heatmap.py) используют softmax centroid |
| Пара heatmap | [crop generator](../../../tools/auto_crop_from_detector.py), [ноутбук](../../../google_colab/train_heatmap_detector.ipynb) | 2 отдельные одноканальные модели; argmax и поосевое масштабирование |

Общий evaluator не воспроизводит оценку пары моделей автоматически. Имена .pth
не задают совместимость. [Регрессионный dataset](../../../training/datasets/tmj_detector_dataset.py)
сортирует InstanceNumber и не применяет HU-rescale; HTTP применяет slope/intercept.
Общий [heatmap dataset](../../../training/datasets/tmj_heatmap_dataset.py) имеет cached
npy/255 и DICOM percentile paths; augmentation использует hardcoded factor 6.

Crop generator сортирует InstanceNumber, resize до 96×128×128, извлекает raw-HU ROI
по умолчанию 128³ с нулевым padding; affine NIfTI единичный. Обрабатывает все
train+val+test либо выбранные study IDs; skip-existing проверяет два файла.
Это генерация входов, не patient-grouped split классификатора.

[2D dataset](../../../training/datasets/tmj_dataset.py) использует HU clip [-1000,2000]
и mask>0.5; [3D dataset](../../../training/datasets/tmj_3d_dataset.py) — min/max и mask>0.
[Evaluator](../../../validation/evaluate_model.py) выполняет slice/max, resize и output>0.5.
В [metrics](../../../validation/metrics.py) Hausdorff в voxels, ASD принимает spacing.
Это текущие технические пороги, а не утверждённые пороги научной приёмки.

[Heatmap model tests](../../../tests/test_tmj_heatmap_detector.py),
[dataset tests](../../../tests/test_tmj_heatmap_dataset.py) и
[loss tests](../../../tests/test_heatmap_loss.py) существуют; здесь не запускались.
