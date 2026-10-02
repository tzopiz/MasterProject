# Трассировка MLService

TC определены рядом с требованиями; этот реестр содержит только ссылки.
«Существует тест» означает наличие tracked проверки, не её успешный запуск.
Тесты в ходе оформления спецификации не запускались; инференс не выполнялся.

| Требования и приёмка | Реализация / существующее покрытие | Статус |
|---|---|---|
| [Требования данных / TC](../functional/data/README.md) | [Label table](../../../training/tmj_position_label_table.py), [тест join](../../../tests/test_tmj_position_label_table.py), [бинаризация](../../../tests/test_binarize_labels.py) | Частичное unit-покрытие; patient identity/provenance открыты |
| [Требования локализации / TC](../functional/localization/README.md) | [Heatmap utilities](../../../training/utils/heatmap.py), [тесты](../../../tests/test_heatmap_utils.py), [генератор ROI](../../../tools/auto_crop_from_detector.py) | Unit для общей heatmap; сквозной ROI/геометрии нет |
| [Требования классификации / TC](../functional/classification/README.md) | [CV](../../../training/sagittal_binary_cv.py), [fold tests](../../../tests/test_stratified_group_kfold.py), [binary metrics tests](../../../tests/test_binary_metrics.py) | Проверки частей; независимое качество не подтверждено |
| [Требования сегментации / TC](../functional/segmentation/README.md) | [2D dataset](../../../training/datasets/tmj_dataset.py), [3D dataset](../../../training/datasets/tmj_3d_dataset.py), [evaluator](../../../validation/evaluate_model.py) | Статическая сверка; выделенных tests не найдено |
| [Нефункциональные требования / TC](../nonfunctional/README.md) | [Протокол](../../../experiments/template/README.md), [app.py](../../../app.py), [обезличивание](../../../tools/dicom_phi_strip.py) | Пробелы воспроизводимости и trust boundary |
| [Требования интеграции / TC](../integration/README.md) | [HTTP](../../../app.py), [detector service](../../../services/detector_service.py) | Статическая сверка; HTTP не запущен |

[Технические записи](../../ai/README.md) фиксируют проверяемые особенности кода;
[пробелы и решения](../open-questions/README.md) связывают их с незакрытой приёмкой.
Наличие shape/backward теста не доказывает качество обученных весов.
