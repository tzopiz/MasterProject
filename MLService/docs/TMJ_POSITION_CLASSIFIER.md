# TMJ Position Classifier

3D CNN модель, предсказывающая положение головок ВНЧС по данным КЛКТ.

Реализована в рамках [issue #67](https://github.com/tzopiz/MasterProject/issues/67).

---

## Задача

Для каждого КЛКТ-объёма модель выдаёт **четыре дискретные метки**:

| Метка | Проекция | Сторона | Классы |
|---|---|---|---|
| `sag_right` | Сагиттальная | Правая | 0 — центральное, 1 — мезиально, 2 — дистально |
| `sag_left`  | Сагиттальная | Левая  | 0 — центральное, 1 — мезиально, 2 — дистально |
| `fr_right`  | Фронтальная  | Правая | 0 — центральное, 1 — медиально, 2 — латерально |
| `fr_left`   | Фронтальная  | Левая  | 0 — центральное, 1 — медиально, 2 — латерально |

Маппинг кодов из `tmj_position_labels.json`:
- Сагиттальные коды 1–3 → класс `код − 1`
- Фронтальные коды 4–6 → класс `код − 4`

---

## Зависимости данных

| Файл | Описание |
|---|---|
| `data/dataset_cbct_public/manifest_private.json` | Таблица соответствия `study_id` ↔ `patient_name`. Содержит ФИО — **не коммитить**. |
| `data/tmj_position_labels.json` | Клинические метки по пациентам (сагитталь/фронталь). |
| `data/dataset_cbct_public/study_*/` | Папки с `.dcm` файлами серий. |

---

## Структура файлов

```
MLService/
├── training/
│   ├── tmj_position_label_table.py       # Stage 1: join manifest + labels
│   └── datasets/
│       └── tmj_position_dataset.py       # Stage 2: PyTorch Dataset + DataLoader
├── models/
│   └── tmj_position_classifier.py        # Stage 3: 3D CNN модель
├── train_tmj_position_classifier.py       # Stage 4: скрипт обучения
└── tests/
    ├── test_tmj_position_label_table.py   # Stage 5: тесты маппинга и индекса
    └── test_tmj_position_classifier_model.py  # Stage 5: тесты формы тензоров
```

---

## Обучение

```bash
cd MLService

./venv/bin/python train_tmj_position_classifier.py \
    --dataset-root    data/dataset_cbct_public \
    --labels-json     data/tmj_position_labels.json \
    --manifest-private data/dataset_cbct_public/manifest_private.json \
    --epochs 100 \
    --batch-size 2 \
    --output-dir experiments/position_run1
```

Все аргументы CLI:

| Аргумент | Умолчание | Описание |
|---|---|---|
| `--dataset-root` | `data/dataset_cbct_public` | Папка с `study_*` |
| `--labels-json` | `data/tmj_position_labels.json` | Файл меток |
| `--manifest-private` | `data/dataset_cbct_public/manifest_private.json` | Манифест |
| `--epochs` | 100 | Число эпох |
| `--batch-size` | 2 | Размер батча |
| `--lr` | 1e-4 | Скорость обучения |
| `--weight-decay` | 1e-5 | L2-регуляризация |
| `--downsample-factor` | 6 | Коэффициент даунсэмплинга |
| `--split-ratio` | 0.8 | Доля train (сплит по пациентам) |
| `--lr-patience` | 10 | Patience для ReduceLROnPlateau |
| `--early-stopping` | 30 | Patience раннего останова |
| `--device` | авто | `cpu` / `cuda` / `mps` |
| `--output-dir` | `experiments` | Корень экспериментов |

Артефакты в `experiments/position_<timestamp>/`:
- `config.json` — конфиг запуска
- `best_model.pth` — лучший чекпоинт (по средней val accuracy)
- `metrics.jsonl` — метрики per-epoch (JSONL)
- `train.log` — лог обучения

---

## Запуск тестов

```bash
cd MLService
./venv/bin/python -m pytest tests/test_tmj_position_label_table.py \
                             tests/test_tmj_position_classifier_model.py -v
```

---

## Архитектура модели

```
TMJPositionClassifier
├── backbone  (4 × ConvBlock3d → MaxPool3d)
│   Conv3d(1→16) → BN → ReLU → Conv3d → BN → ReLU → MaxPool3d(2)
│   Conv3d(16→32) → … → MaxPool3d(2)
│   Conv3d(32→64) → … → MaxPool3d(2)
│   Conv3d(64→128) → … → MaxPool3d(2)
├── AdaptiveAvgPool3d(1)  →  (B, 128)
├── head_sag_right  Linear(128→256) → ReLU → Dropout → Linear(256→3)
├── head_sag_left   (то же)
├── head_fr_right   (то же)
└── head_fr_left    (то же)
```

Loss: `CrossEntropyLoss` × 4, суммируется.
Метрика сохранения чекпоинта: **средняя accuracy** по всем четырём головам на val.

---

## Ограничения MVP (итерация 1)

> **Центральный кроп вместо ROI-детектора.**
>
> Модель принимает фиксированный центральный кроп фиксированного размера
> (по умолчанию 96×128×128 вокселей после даунсэмплинга ×6) из всего объёма.
> Это убирает зависимость от детектора, но снижает точность, так как кроп
> может не захватить обе головки при нестандартном положении пациента.

### Переход ко второй итерации (отдельный этап #67)

1. Использовать выходы `TMJDetector` (координаты головок) для вычисления
   индивидуальных кропов вокруг левой и правой головки.
2. Передать два кропа в две отдельные ветки энкодера или использовать один
   общий энкодер с разными кропами.
3. Убрать `DEFAULT_CROP` как гиперпараметр — размер кропа задаётся детектором.
