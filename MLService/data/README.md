# MLService / data

Локальные данные для обучения и инструментов: DICOM-серии, кропы, метки, выгрузки когорты.

## Типичная структура (по мере использования)

| Путь | Назначение |
|------|------------|
| `dataset/` | Упорядоченные серии после `tools/organize_dataset.py`. |
| `dataset_cbct_public/` | Публичная когорта после подготовки (см. корневой ML README). |
| `detector_crops*`, `auto_crops/`, `processed_crops/` | Кропы под классификатор и сегментацию. |
| `tmj_position_labels.json` | Метки положения головок (коды 1–6). |
| `heatmap_volumes/` | Промежуточные объёмы для heatmap-детектора. |

## Git и приватность

- Каталоги **`cbct_public_zips/`** и **`cbct_public_extracted/`** указаны в корневом `.gitignore` (большие архивы).
- Файл **`manifest_private.json`** (ФИО ↔ пути) не должен попадать в git — см. `.gitignore`.

Скрипты для загрузки и очистки когорты: [../tools/README.md](../tools/README.md).
