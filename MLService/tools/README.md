# MLService / tools

Скрипты и небольшие приложения для датасетов, разметки, кропов и визуализации. Запуск из каталога **`MLService/`** (чтобы пути `data/`, `tools/` совпадали).

## Документы

- **[README_ROI_TOOL.md](README_ROI_TOOL.md)** — разметка центров суставов (`roi_annotation_tool.py`).
- **[tmj_classification_tool/](tmj_classification_tool/)** — веб-инструмент классификации кропов; см. [tmj_classification_tool/README.md](tmj_classification_tool/README.md) и [tmj_classification_tool/QUICKSTART.md](tmj_classification_tool/QUICKSTART.md).

## Группы скриптов (ориентир)

| Направление | Примеры файлов |
|-------------|----------------|
| Датасет и когорта | `organize_dataset.py`, `prepare_cbct_cohort.py`, `download_yandex_cbct_cohort.py`, `build_cbct_zip_dataset.py`, `sync_cbct_cohort.py` |
| Детектор и кропы | `auto_crop_from_detector.py`, `visualize_detector.py`, `extract_crops_from_annotations.py` |
| Разметка положения | `parse_tmj_position_labels_docx.py`, `batch_annotate.py` |
| Heatmap / оценка | `preprocess_heatmap_volumes.py`, `evaluate_detector.py` |
| Визуализация | `visualize_crops.py`, `visualize_3d.py`, `plot_merged_hu_histograms.py` |
| Прочее | `anonymize_labels.py`, `dicom_phi_strip.py`, `create_portable_tool.py` |

Полный сценарий пайплайна и примеры команд: [../README.md](../README.md).
