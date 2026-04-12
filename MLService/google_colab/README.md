# TMJ Position Classifier — обучение в Colab / DataSphere / локально

Ноутбуки для обучения классификатора положения головок ВНЧС по КЛКТ-кропам. Полная сводка экспериментов, метрик, сбоев и обходов — в **[POSITION_CLASSIFIER_EXPERIMENTS.txt](POSITION_CLASSIFIER_EXPERIMENTS.txt)** (обновляйте при новых прогонах и `training_analysis*.json`).

## Ноутбуки

| Файл | Назначение |
|------|------------|
| [train_position_classifier.ipynb](train_position_classifier.ipynb) | 3D CNN на кропах 128³ (v3: балансировка, аугментации, weighted CE). Нужны DICOM + детектор + препроцессинг кропов. |
| [train_position_classifier_2d.ipynb](train_position_classifier_2d.ipynb) | 2D multi-view (ResNet18): Approach C/A + секция **4b v5** (бинарные метки, multi-slice, CV по пациентам, SVM). Достаточно `tmj_crops/*.npy` + метки + manifest. |
| [train_binary_position_classifier.ipynb](train_binary_position_classifier.ipynb) | **Approach A+B** — бинарная классификация (central vs non-central) с детектор-кропами 128³ (NIfTI) и BinaryFocalLoss + калибровка порога. Полностью самодостаточный ноутбук, поддерживает DataSphere (V100), Colab и локально. |
| [train_sagittal_binary_cv.ipynb](train_sagittal_binary_cv.ipynb) | **Сагитталь, CV:** обёртка над `training/sagittal_binary_cv.py`. **DataSphere:** данные из датасета `tmj` (`/home/jupyter/datasets/tmj/`, кропы `detector_crops_v2`), пути подставляет `training/utils/datasphere_env.py`; артефакты JSON — в `filestore/experiments/`. Инициализация датасета с GitHub: [init_datasphere_dataset.ipynb](init_datasphere_dataset.ipynb). |

### Артефакты binary-ноутбука

Все файлы прогона (веса, `metrics.jsonl`, графики, `training_analysis.json`, ZIP) — в **`MLService/experiments/sag_only_<timestamp>/`**. Примеры локальных копий: **`../experiments/sag_only_20260411_191537/`** (последний зафиксированный), **`../experiments/sag_only_20260411_182037/`** — в каждой папке см. `README.md`. Сводка экспериментов: [POSITION_CLASSIFIER_EXPERIMENTS.txt](POSITION_CLASSIFIER_EXPERIMENTS.txt).

## Данные

### Полный пайплайн (3D-ноутбук)

На Drive / в `tmj_data/`:

```
tmj_data/
├── dataset_public/
│   ├── manifest_private.json
│   └── study_XXXX/   # DICOM
├── tmj_position_labels.json
├── models/best_detector.pth   # для препроцессинга кропов
└── tmj_crops/                 # после preprocess в 3D-ноутбуке
```

### Облегчённый набор (2D-ноутбук)

Достаточно **lite**-архива или папки:

```
tmj_data/
├── tmj_position_labels.json
├── dataset_public/manifest_private.json
└── tmj_crops/*.npy
```

## Где лежат данные (по средам)

- **Google Colab:** монтирование Drive → `My Drive/tmj_data/` (ячейка с `drive.mount`).
- **Yandex DataSphere:** датасет проекта `tmj_data` → `/home/jupyter/datasets/tmj_data/`; кэш фич и артефакты — `/home/jupyter/project/` (см. ячейки Setup в 2D-ноутбуке). Прямое скачивание с Google Drive из среды часто **недоступно** — типичный путь: скачать `tmj_data_lite.zip` локально, **Upload** в проект, `unzip` в `datasets/`, при необходимости оформить датасет через `#pragma dataset init` (см. [POSITION_CLASSIFIER_EXPERIMENTS.txt](POSITION_CLASSIFIER_EXPERIMENTS.txt)).
- **Локально:** `./data/tmj_data/` относительно ноутбука.

## Запуск

1. **2D:** `%pip` в первой ячейке; выполнять Setup → данные → далее по секциям. GPU желателен для ResNet-части; v5 (SVM) нормально на CPU.
2. **3D:** GPU в Colab; после `pip install` — перезапуск сессии; см. комментарии в ноутбуке про DICOM (pylibjpeg).

## Форматы меток и manifest

Как в исходной версии README (см. ниже) — `manifest_private.json` со списком `studies`, `tmj_position_labels.json` с `patients` и кодами сагиттали/фронтали.

**`manifest_private.json`:**

```json
{
  "studies": [
    {"study_id": "study_0001", "patient_name": "Иванов Иван Иванович"},
    {"study_id": "study_0002", "patient_name": "Петров Пётр Петрович"}
  ]
}
```

**`tmj_position_labels.json`:**

```json
{
  "schema_version": "1.0",
  "patients": [
    {
      "name_raw": "Иванов Иван Иванович",
      "labels": {
        "sagittal": {"right": 1, "left": 2},
        "frontal":  {"right": 4, "left": 6}
      }
    }
  ]
}
```

## Связь с сервисом

Обучение «в репозитории» для продакшена описано в [MLService/README.md](../README.md) (`train_tmj_position_classifier.py`, [docs/README.md](../docs/README.md)). Ноутбуки здесь — исследовательский контур (Colab/DataSphere); веса и отчёты выгружаются вручную.
