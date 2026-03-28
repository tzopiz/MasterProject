# Публичная CBCT-когорта: метки из DOCX, Яндекс.Диск, сбор датасета

Документ фиксирует **пайплайн, скрипты и артефакты**, добавленные для подготовки данных к задаче **классификации положения головок ВНЧС** (коды 1–6 в сагиттали и фронтали, слева/справа).

## Зачем это нужно

После выполнения шагов локально получаются:

1. **`MLService/data/tmj_position_labels.json`** — разбор клинического DOCX: ФИО пациента и **четыре числовые метки** на человека (`labels.sagittal` / `labels.frontal`, поля `right` / `left`).
2. **`MLService/data/cbct_public_zips/*.zip`** — исследования с **публичной папки Яндекс.Диска** (сопоставление zip ↔ пациент из JSON по ФИО, fuzzy).
3. **`MLService/data/cbct_public_extracted/<имя архива>/`** — распакованные деревья с **DICOM** после удаления мусора (вьюеры, логи, превью и т.д.).

Если данные уже лежат в других каталогах, переименуйте папки под эти имена или укажите свои пути флагами `--zips-dir`, `--extract-dir`, `--dataset-out` / `--input`, `--output`.

Дальше по ML-пайплайну обычно идут `organize_dataset.py` (желательно с **`--anonymize`** для выкладки), детектор ROI, кропы и обучение классификатора (см. [MLService/README.md](../MLService/README.md)).

## Скрипты (`MLService/tools/`)

| Скрипт | Роль |
|--------|------|
| `parse_tmj_position_labels_docx.py` | DOCX → JSON (`schema_version`, `class_legend`, `patients[]`, `errors[]`). |
| `download_yandex_cbct_cohort.py` | Список zip через API Диска, матчинг к `name_raw`, скачивание в `--output-dir`. |
| `prepare_cbct_cohort.py` | Опционально загрузка + распаковка + очистка (оркестратор). |
| `build_cbct_zip_dataset.py` | Только распаковка всех zip из папки + очистка (удобно, если zip уже есть). |
| `sync_cbct_cohort.py` | **Один прогон:** распаковка новых zip → `organize_dataset` (по умолчанию с **анонимизацией DICOM**). |
| `organize_dataset.py` | Плоская структура `study_0001`…, `manifest.json`; флаг **`--anonymize`** — снятие PHI в DICOM + публичный manifest без ФИО/путей + `manifest_private.json` (локально, в `.gitignore`). |
| `dicom_phi_strip.py` | Служебный модуль для снятия тегов PHI при анонимизации. |
| `dicom_cohort_cleanup.py` | Общие правила «мусора» и `safe_extract_zip` (zip-slip); используется скриптами распаковки. |

### Важные флаги загрузчика

- **`--dry-run`** — план без скачивания.
- **`--download-below-threshold`** — скачать лучший матч даже при низком `token_sort_ratio` относительно `--min-score` (по умолчанию 0.82). В манифесте помечается как `downloaded_low_confidence` / предупреждение.
- **`--strict-download-match`** у `prepare_cbct_cohort.py` — не передавать `download-below-threshold` (строгий режим).

## Команды (из каталога `MLService/`)

```bash
# 1. Метки из DOCX
python3 tools/parse_tmj_position_labels_docx.py -i /path/to/клиника.docx -o data/tmj_position_labels.json --pretty

# 2. Скачивание (сначала dry-run)
python3 tools/download_yandex_cbct_cohort.py \
  --labels data/tmj_position_labels.json \
  --output-dir data/cbct_public_zips \
  --dry-run

python3 tools/download_yandex_cbct_cohort.py \
  --labels data/tmj_position_labels.json \
  --output-dir data/cbct_public_zips \
  --download-below-threshold

# 3. Распаковка + очистка под датасет
python3 tools/build_cbct_zip_dataset.py --dry-run
python3 tools/build_cbct_zip_dataset.py

# Или полный цикл «скачать → распаковать → почистить»
python3 tools/prepare_cbct_cohort.py
python3 tools/prepare_cbct_cohort.py --no-download

# Распаковка (новые zip) + организованный датасет с анонимизацией DICOM
python3 tools/sync_cbct_cohort.py
python3 tools/sync_cbct_cohort.py --download

# Только организация (например, свой --output)
python3 tools/organize_dataset.py --input data/cbct_public_extracted --output data/dataset_cbct_public --anonymize
```

Публичный URL папки на Диске задаётся в скриптах флагом `--public-url` (значение по умолчанию — то, что использовалось при разработке пайплайна).

## Файлы-отчёты

| Файл | Содержание |
|------|------------|
| `data/cbct_public_zips/download_manifest.json` | Для каждого `patient_number`: `matched_zip`, `score`, `status`, пути, предупреждения. |
| `data/cbct_public_extracted/prepare_report.json` | Результат `prepare_cbct_cohort.py` (фазы download / extract / clean). |
| `data/cbct_public_extracted/dataset_build_report.json` | Результат `build_cbct_zip_dataset.py`. |

## Схема меток в `tmj_position_labels.json`

- **`class_legend`**: тексты кодов 1–6 из шапки DOCX.
- **`patients[]`**: `patient_number`, `name_raw`, поля шапки (`visit_raw`, даты в сыром виде), **`labels`**:  
  `sagittal: { right, left }`, `frontal: { right, left }` — целые **1–6** (фронталь в клинике обычно 4–6).

Связка с папкой на диске: имя **`cbct_public_extracted/<stem>/`** совпадает с именем **`.zip`** (без расширения), которое в свою очередь сопоставлено с `name_raw` при загрузке. Для сопоставления **`study_xxxx` ↔ клинические метки** после `--anonymize` используйте локальный **`manifest_private.json`** (не коммитить).

## Ограничения и качество данных

- **Пациент №2 (Якунин):** в DOCX могло быть написано «Артём» — **опечатка**; в `tmj_position_labels.json` для сопоставления с архивом на Яндекс.Диске имя приведено к **«Якунин Марк Викторович»** (см. `parse_notes` у записи). После перегенерации JSON из DOCX правку нужно повторить или исправить исходный DOCX.
- При **обновлении DOCX** число пациентов в JSON может вырасти — для **новых** номеров нужно снова запустить загрузчик с Диска (появятся новые `.zip` или статусы в манифесте).
- Несколько zip на одного человека встречаются редко; при дублях выбирается лучший fuzzy-match — проверяйте `score` в манифесте.
- Каталоги **`cbct_public_zips/`** и **`cbct_public_extracted/`** указаны в **корневом `.gitignore`** (большой объём); в репозитории остаются **скрипты** и при необходимости **`tmj_position_labels.json`**.

## Где ещё описано

- Список инструментов и примеры команд: [MLService/README.md](../MLService/README.md) (раздел Tools).
- Правила очистки дублируют идеи [file_cleaner](../MLService/tools/tmj_classification_tool/services/file_cleaner.py), но **расширены** в `dicom_cohort_cleanup.py` (логи, html, `__MACOSX` и т.д.).
