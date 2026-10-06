# TMJ Analysis ML Service

Спецификация поведения: [правила и приёмка](docs/spec/README.md). Текущая реализация и доказательства: [инженерный справочник](docs/ai/README.md). Общий [порядок разработки](../docs/README.md) применяется к новым изменениям.

Сервис для автоматического анализа КЛКТ (CBCT) снимков ВНЧС (TMJ).
Этот проект предоставляет API для анализа DICOM файлов, а также инструменты для обучения моделей детекции и сегментации.

## Документация по папкам

Краткие входы: [models/README.md](models/README.md) · [tools/README.md](tools/README.md) · [training/README.md](training/README.md) · [services/README.md](services/README.md) · [utils/README.md](utils/README.md) · [data/README.md](data/README.md) · [tests/README.md](tests/README.md) · [docs/README.md](docs/README.md) · [validation/README.md](validation/README.md) · [experiments/README.md](experiments/README.md) · [google_colab/README.md](google_colab/README.md)

## Исследовательский путь перед новым обучением

Основной вход — [единый config, проверка данных и запуск](google_colab/README.md); [DataSphere: подготовить → подтвердить → получить результат](docs/cloud-launch/README.md) описывает перенос данных и платное задание. Контракт всего пути и критерии проверки находятся в [спецификации подготовки](docs/spec/features/dataset-readiness/README.md).

- Исходный файл связывает конкретные исследования, пациентов и метки; [строгий индекс](docs/spec/functional/data/README.md) не угадывает соответствия по имени или номеру случая.
- Для обучения используются две проверенные ROI на исследование с паспортами происхождения. Из DICOM их создаёт [paired heatmap pipeline](docs/ai/data/README.md); веса исторического регрессионного API не заменяют эти локализаторы.
- [DICOM-инференс](docs/ai/inference/README.md) использует завершённый прогон всех folds и те же локализаторы/ROI preprocessing; доступны binary и sagittal multiclass.
- Параметры задаются одним приватным JSON. Preflight выполняется до обучения; сохранённые checkpoints, предсказания и параметры позволяют проверить результат. Validation участвует в выборе модели, поэтому отчёт обозначен `development_cv`.

```bash
# Из MLService, Python 3.12; пути в config относятся к его каталогу.
python -m tools.run_research --config /private/staging/research.private.json --preflight
python -m tools.run_research --config /private/staging/research.private.json
```

Файл исследований и ссылку на разрешённый датасет можно передать для подготовки конкретного запуска. Недостающая идентичность пациента, применимость метки или совместимость весов остаются причиной отказа, а не заполняются догадкой. Исходные DICOM, таблица соответствий и построчные результаты остаются приватными; псевдонимизация не доказывает анонимность пикселей. Открытые источники для отдельной разметки: [проверенный список и ограничения](docs/public-datasets/README.md).

Ниже описаны исторические инструменты и API. Их команды и результаты не заменяют основной исследовательский config и не доказывают качество нового обучения.

---

## Исторические инструменты пайплайна

Весь процесс разделен на 3 основных этапа:

### 1. Подготовка данных (Dataset)
Организация сырых DICOM файлов в структурированный вид.
- **Инструмент:** `tools/organize_dataset.py`
- **Вход:** Папка с сырыми DICOM файлами.
- **Выход:** Структура `data/dataset/study_XXXX/`.

```bash
python tools/organize_dataset.py --input <raw_data_folder> --output data/dataset
```

### 2. Детекция (Localization)
Нахождение координат центра сустава (ROI - Region of Interest) на полном 3D снимке.
- **Разметка:** `tools/roi_annotation_tool.py` (GUI для клика по центрам суставов).
- **Обучение:** `train_detector.py` (Обучает модель находить координаты).
- **Инференс:** `tools/auto_crop_from_detector.py` (Использует обученную модель для вырезания кропов 128x128x128).

**Результат:** JSON файлы с координатами и вырезанные 3D кубы суставов в `data/auto_crops/`.

### 3. Классификация положения (Position Classification)
Предсказание положения головок ВНЧС (сагитталь + фронталь, лево + право).
- **Метки:** `data/tmj_position_labels.json` (6 кодов, 4 метки на пациента).
- **Документация:** [docs/README.md](docs/README.md)
- **Colab / DataSphere / эксперименты:** [google_colab/README.md](google_colab/README.md) (в т.ч. артефакты **CV сагиттали**: `sagittal_cv_last.json`, `*_epochs.jsonl`, `*_analyze_*`); длинная сводка прогонов — [google_colab/POSITION_CLASSIFIER_EXPERIMENTS.txt](google_colab/POSITION_CLASSIFIER_EXPERIMENTS.txt)

#### 3-классовый (v1–v4, baseline)
- **Обучение:** `train_tmj_position_classifier.py`
- **Модель:** `models/tmj_position_classifier.py` — 4 головы × 3 класса (central / anterior / posterior)

```bash
./venv/bin/python train_tmj_position_classifier.py \
    --dataset-root data/dataset_cbct_public \
    --labels-json data/tmj_position_labels.json \
    --manifest-private data/dataset_cbct_public/manifest_private.json
```

#### Бинарный (v6, Approach A+B)
- **Обучение:** `train_binary_position_classifier.py` или ноутбук `google_colab/train_binary_position_classifier.ipynb`
- **Модель:** `models/tmj_binary_position_classifier.py` — 2 головы × 1 логит (central vs non-central)
- **Loss:** `training/losses/focal_loss.py` — `BinaryFocalLoss(γ=2, α=auto)`
- **Подход A:** NIfTI-кропы от детектора (128³) вместо центрального кропа DICOM
- **Подход B:** `BinaryFocalLoss` + калибровка порога по Youden's J на val ROC
- **Артефакты ноутбука:** `experiments/sag_only_<timestamp>/` (веса, `metrics.jsonl`, `training_analysis.json`, графики); в §9 — ZIP `*_bundle.zip` для скачивания. Примеры полных копий: [`experiments/sag_only_20260411_191537/README.md`](experiments/sag_only_20260411_191537/README.md) (последний зафиксированный прогон), [`experiments/sag_only_20260411_182037/README.md`](experiments/sag_only_20260411_182037/README.md); индекс — [`experiments/README.md`](experiments/README.md).

```bash
# Исторический trainer принимает уже подготовленные кропы.
# Их canonical генерация и паспорта описаны в docs/ai/data/README.md.
# Для нового исследования используйте tools.run_research выше.
# Обучение историческим trainer:
./venv/bin/python train_binary_position_classifier.py \
    --crop-dir data/detector_crops \
    --labels-json data/tmj_position_labels.json \
    --manifest-private data/dataset_cbct_public/manifest_private.json \
    --dataset-root data/dataset_cbct_public

# Шаг 3: проверить кропы визуально
./venv/bin/python tools/visualize_crops.py
```

### 4. Сегментация (Segmentation)
Точное выделение костных структур внутри вырезанного кропа.
- **Разметка:** Ручная разметка кропов в **ITK-SNAP**.
- **Обучение:** `train_3d.py` (Обучает 3D U-Net на кропах).
- **Модель:** `models/unet_3d.py`.

---

## 🛠 Установка и Запуск

### Требования
- Python 3.9+
- macOS / Linux (рекомендуется)
- CUDA (опционально, для ускорения обучения)

### Быстрый старт
```bash
cd MLService
./start.sh
```

Или вручную:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Сервис будет доступен по адресу: `http://localhost:8001`

---

## 🧠 Обучение моделей

### TMJ Detector (Этап 2)
Обучение модели для поиска суставов на полном снимке.

```bash
# 1. Разметка данных (создает JSON файлы)
python tools/roi_annotation_tool.py

# 2. Обучение модели
python train_detector.py --epochs 200 --batch_size 2

# 3. Мониторинг (в другом терминале)
tail -f experiments/detector_*/training.log
```

### 3D Segmentation (Этап 3)
Обучение модели сегментации на подготовленных кропах.

```bash
# 1. Подготовка кропов (используя обученный детектор)
# Paired heatmap ROI: см. docs/ai/data/README.md; regression weights несовместимы.

# 2. Обучение 3D U-Net
python train_3d.py --data_dir data/auto_crops --epochs 100
```

---

## 📡 API Reference

Сервис предоставляет REST API для интеграции с Backend.

### 1. Health Check
`GET /health`
Проверка работоспособности сервиса и статуса загрузки модели.

### 2. Start Analysis
`POST /process`
Запуск анализа DICOM файла.
**Body:**
```json
{
  "dicom_path": "/path/to/file.dcm",
  "task_id": "uuid-string"
}
```

### 3. Check Model Status
`GET /models/status`
Информация о загруженной модели (тип, путь).

---

## 🛠 Инструменты (Tools)

В папке `tools/` находятся утилиты для работы с данными:

- `roi_annotation_tool.py`: GUI приложение для быстрой разметки центров суставов.
- `organize_dataset.py`: Скрипт для сортировки и переименования DICOM; флаг **`--anonymize`** — снятие PHI и публичный `manifest.json` без ФИО.
- `auto_crop_from_detector.py`: Автоматическое создание кропов с использованием детектора.
- `manual_crop_tool.py`: (Legacy) Ручное создание кропов.
- `create_portable_tool.py`: Создание portable-версии разметчика для передачи врачам/коллегам.
- `visualize_detector.py`: Визуальная проверка работы детектора (рисует bounding box на снимке).
- `extract_crops_from_annotations.py`: Создание кропов на основе JSON аннотаций (без детектора).
Полный сценарий **публичной CBCT-когорты** (DOCX, Яндекс.Диск, zip, датасет) — см. скрипты в `tools/` и разделы про данные в этом README.

- `parse_tmj_position_labels_docx.py`: Разбор DOCX с кодами 1–6 и блоками «Пациент N. …» в JSON (разметка положения головок ВНЧС).

```bash
python tools/parse_tmj_position_labels_docx.py -i путь/к/файлу.docx -o labels.json --pretty
```

- `download_yandex_cbct_cohort.py`: скачивание `.zip` пациентов из публичной папки Яндекс.Диска по `tmj_position_labels.json` (сначала `--dry-run`).

```bash
python tools/download_yandex_cbct_cohort.py --labels data/tmj_position_labels.json \\
  --output-dir data/cbct_public_zips --dry-run
```

- `prepare_cbct_cohort.py`: скачать все zip (по умолчанию с `--download-below-threshold`), распаковать в `data/cbct_public_extracted/<имя>/`, удалить мусор.

```bash
python tools/prepare_cbct_cohort.py
python tools/prepare_cbct_cohort.py --no-download
```

- `build_cbct_zip_dataset.py` — только **распаковка zip → папка пациента** и **расширенная очистка** (логи, html, вьюеры, `__MACOSX`, пустые каталоги). Правила: `tools/dicom_cohort_cleanup.py`.

```bash
python tools/build_cbct_zip_dataset.py --dry-run
python tools/build_cbct_zip_dataset.py
```

- `sync_cbct_cohort.py` — распаковка новых zip + **`organize_dataset --anonymize`** (единая команда для обновления датасета).

```bash
python tools/sync_cbct_cohort.py
python tools/sync_cbct_cohort.py --download
```

---

## 📊 Мониторинг обучения

Для Detector / Segmentation смотрите вывод процесса в терминале и каталог прогона `experiments/<имя>/` (`train.log`, `metrics.jsonl`, `best_model.pth`, `config.json`).

**Ориентиры:**
- `ls -la experiments/detector_*/` — какие прогоны есть.
- `tail -f experiments/<прогон>/train.log` — если лог пишется в файл (зависит от скрипта).
- `grep -E "Epoch|Saved best|MAE|Early"` — по сохранённому логу или выводу в терминале.
- `cat experiments/<прогон>/config.json` — параметры запуска.

---

## 📦 Deployment (Развертывание)

### Docker
Создайте `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Переменные окружения (Production)
- `MODEL_PATH`: Путь к файлу модели (.pth).
- `API_PORT`: Порт сервиса (default: 8001).
- `WORKERS`: Количество воркеров uvicorn.

---

## 📚 Legacy & Manual Workflows

### Ручная сегментация (ITK-SNAP)
Для создания Ground Truth масок:
1. Откройте кроп (`study_XXXX_left.nii.gz`) в ITK-SNAP.
2. Создайте сегментацию (Label 1).
3. Сохраните как `study_XXXX_left_seg.nii.gz`.

### Анализ серии (Legacy Script)
Ранее использовался скрипт `analyze_dicom_series.py` для генерации превью (axial/coronal/sagittal) из папки с DICOM. Сейчас эта логика частично интегрирована в основной пайплайн или заменена инструментами визуализации.

Первый бинарный osseous GPU run 2026-10-06 завершён, артефакты проверены.
Качество ниже constant baseline (test AUROC 0.291 против 0.500); это
исследовательский отрицательный результат по авторским ROI. Подробности,
границы оценки и следующий этап — [протокол](experiments/tmj_od3d_feasibility_20261005/README.md#результат-первого-binary-training-2026-10-06).

Osseous research config теперь допускает экспериментальный `architecture="spatial_head"`;
default `global_mean` сохраняет прежнее поведение. Вариант ещё не выбран CV;
обобщение неизвестно. Контракт и приёмка: [архитектура](docs/spec/functional/osseous/README.md#fr-ml-osseous-architecture).
