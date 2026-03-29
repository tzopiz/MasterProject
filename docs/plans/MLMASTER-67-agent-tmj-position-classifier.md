# План для агента: классификатор положения головок ВНЧС (GitHub #67)

**Трекинг:** все коммиты и PR связывать с [issue #67](https://github.com/tzopiz/MasterProject/issues/67) (комментарий к issue или `Refs #67` в описании PR).

**Задача:** реализовать обучаемую модель, которая по объёму КЛКТ (или кропу) предсказывает **четыре дискретные метки** из `tmj_position_labels.json`: для каждой стороны (лево/право) отдельно класс в **сагиттали** (коды 1–3) и в **фронтали** (коды 4–6). На выходе четыре головы с **3 классами** каждая: сагиттальные метки маппить в `{0,1,2}` как `код − 1`, фронтальные — как `код − 4`. Потери — сумма четырёх `CrossEntropyLoss`. Первая версия (MVP) — **центральный кроп** фиксированного размера из всего объёма серии (без детектора), чтобы снять зависимость от ROI; документировать ограничение в README модуля.

**Структура плана:** подготовка таблицы примеров → датасет PyTorch → модель → скрипт обучения и метрики → минимальные тесты → краткая документация.

---

### Stage 1: Таблица соответствия `study_*` ↔ метки пациента

**Что добавить/реализовать:**

*   Модуль или скрипт, который читает `data/dataset_cbct_public/manifest_private.json` (поля `study_id`, `patient_name`) и `data/tmj_position_labels.json` (поля `patients[].name_raw`, `patients[].labels.sagittal|frontal` с `left`/`right`), сопоставляет по **точному совпадению** `patient_name` и `name_raw`.
*   Строит список записей: `study_id`, путь к папке с `.dcm`, четыре целевых индекса `sag_right`, `sag_left`, `fr_right`, `fr_left` в диапазоне 0..2 (после маппинга описанного выше).
*   Пациенты без совпадения в JSON — исключить из обучения с логированием числа отброшенных `study_*`.
*   **Сплит train/val строго по `patient_name`**, а не по `study_id`, чтобы несколько серий одного человека не попадали в разные сплиты.

**Файлы создать/изменить:**

*   `MLService/training/tmj_position_label_table.py` — функции загрузки JSON, join, маппинг кодов, возврат списка словарей и/или сохранение опционального `data/tmj_position_training_index.json` (кэш для отладки).

**Примеры в существующем коде:**

*   `MLService/training/datasets/tmj_detector_dataset.py` — паттерн загрузки аннотаций и сплита.
*   `MLService/data/tmj_position_labels.json` — схема полей.

**Проверка:**

*   `cd MLService && ./venv/bin/python -c "from training.tmj_position_label_table import build_index; x=build_index(); print(len(x), x[0].keys())"` — выполняется без исключения, длина > 0.

---

### Stage 2: Класс датасета `TMJPositionClassificationDataset`

**Что добавить/реализовать:**

*   `Dataset`, принимающий список записей из Stage 1 и корень `dataset_cbct_public`.
*   Загрузка серии: обход `*.dcm` в `study_id`, сортировка срезов по `ImagePositionPatient[2]` как в `DICOMProcessor.load_series` / `tmj_detector_dataset`.
*   Нормализация HU: как в `TMJDetectorDataset` (clip и масштабирование в float tensor).
*   Даунсэмплинг объёма (коэффициент как гиперпараметр, по умолчанию согласовать с детектором, например 6) затем **центральный crop** до фиксированного `(D,H,W)` (например 96×128×128), при нехватке размера — padding нулями после минимального resize.
*   `__getitem__` возвращает `(volume_tensor, target_dict)` с четырьмя тензорами-метками длиной 3 классов (индексы 0..2) или один тензор формы `(4,)`.
*   Лёгкая аугментация только на train: случайный сдвиг crop до ±N вокселей по осям, без искажения меток.

**Файлы создать/изменить:**

*   `MLService/training/datasets/tmj_position_dataset.py` — класс датасета и фабрика `get_position_dataloaders(...)`.

**Примеры в существующем коде:**

*   `MLService/training/datasets/tmj_detector_dataset.py` — чтение DICOM, downsample, кэш.
*   `MLService/services/dicom_processor.py` — согласование HU.

**Проверка:**

*   `cd MLService && ./venv/bin/python -c "from training.datasets.tmj_position_dataset import TMJPositionClassificationDataset; from training.tmj_position_label_table import build_index; idx=build_index(); ds=TMJPositionClassificationDataset(idx[:3], Path('data/dataset_cbct_public'), is_train=False); x,y=ds[0]; print(x.shape, y)"` — печатает форму тензора и метки.

---

### Stage 3: Модель `TMJPositionClassifier`

**Что добавить/реализовать:**

*   Один энкодер 3D CNN (несколько блоков Conv3d + BN + ReLU + pool), общий backbone.
*   Четыре независимых линейных головы на pooled-признак: каждая выдаёт логиты формы `(batch, 3)`.
*   Метод `forward` возвращает словарь или кортеж из четырёх тензоров логитов в фиксированном порядке: `(sag_right, sag_left, fr_right, fr_left)`.

**Файлы создать/изменить:**

*   `MLService/models/tmj_position_classifier.py` — класс модели и опционально фабрика по размеру входа.

**Примеры в существующем коде:**

*   `MLService/models/tmj_detector.py` — стиль блоков и инициализация.

**Проверка:**

*   `cd MLService && ./venv/bin/python -c "import torch; from models.tmj_position_classifier import TMJPositionClassifier; m=TMJPositionClassifier(); x=torch.randn(2,1,32,48,48); o=m(x); print([t.shape for t in o])"` — четыре выхода `(2,3)`.

---

### Stage 4: Скрипт обучения `train_tmj_position_classifier.py`

**Что добавить/реализовать:**

*   Аргументы CLI: `--dataset-root`, `--labels-json`, `--manifest-private`, `--epochs`, `--batch-size`, `--lr`, `--split-ratio`, `--device`, `--output-dir` (каталог эксперимента с `config.json`, `best_model.pth`, `metrics.jsonl`).
*   Цикл обучения по образцу `train_detector.py`: оптимизатор Adam, scheduler по желанию, логирование в консоль и файл.
*   Метрики на val: accuracy по каждой из четырёх голов, средняя accuracy, сохранение лучшего чекпоинта по средней accuracy.
*   После эпохи — короткий отчёт (можно без полноценного confusion matrix в MVP, опционально сохранить `sklearn.metrics.confusion_matrix` в JSON для одной головы).

**Файлы создать/изменить:**

*   `MLService/train_tmj_position_classifier.py` — точка входа.

**Примеры в существующем коде:**

*   `MLService/train_detector.py` — структура main, логирование, сохранение.

**Проверка:**

*   `cd MLService && ./venv/bin/python train_tmj_position_classifier.py --epochs 1 --batch-size 1 --output-dir experiments/position_smoke` — завершается без traceback; в `experiments/position_smoke` появляются артефакты.

---

### Stage 5: Юнит-тесты датасета и модели

**Что добавить/реализовать:**

*   Тест с **синтетическим** мини-объёмом (случайный tensor + фиктивные метки), проверка формы батча через модель.
*   Тест маппинга кодов 1–3 и 4–6 в 0–2 без датасета диска (pure function).

**Файлы создать/изменить:**

*   `MLService/tests/test_tmj_position_label_table.py`
*   `MLService/tests/test_tmj_position_classifier_model.py`

**Примеры в существующем коде:**

*   Искать `test_*.py` в `MLService/`; если тестов мало — ориентироваться на `pytest` и минимальный assert.

**Проверка:**

*   `cd MLService && ./venv/bin/python -m pytest tests/test_tmj_position_label_table.py tests/test_tmj_position_classifier_model.py -q`

---

### Stage 6: Документация для #67

**Что добавить/реализовать:**

*   Раздел в `MLService/README.md` или отдельный `MLService/docs/TMJ_POSITION_CLASSIFIER.md`: зависимости от `manifest_private.json` + `tmj_position_labels.json`, команда обучения, ограничение MVP (центральный кроп), как перейти на кропы детектора во второй итерации.
*   Комментарий в issue #67 со ссылкой на этот план и на merge-коммит/PR после выполнения.

**Файлы создать/изменить:**

*   `MLService/README.md` или `MLService/docs/TMJ_POSITION_CLASSIFIER.md` (один файл по выбору репозитория; если создаётся отдельный md — добавить ссылку из README).

**Проверка:**

*   Ручная проверка: файл существует, команды из документа копируются в терминал без ошибок путей.

---

## Ограничения, которые агент не оспаривает в рамках плана

*   Путь к датасету по умолчанию: `MLService/data/dataset_cbct_public`.
*   PHI-манифест: `MLService/data/dataset_cbct_public/manifest_private.json` (локально, не в git).
*   Метки: `MLService/data/tmj_position_labels.json`.

После MVP следующий этап (отдельный комментарий к #67, не в этом плане): подставить кропы из `roi_annotations` / детектора вместо центрального crop.
