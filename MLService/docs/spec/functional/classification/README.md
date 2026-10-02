# Классификация положения головок

Многоклассовая задача различает три положения отдельно для каждой плоскости и
стороны; бинарная объединяет только два класса смещения в non-central.
Исследовательская классификация пока не входит в HTTP-результат анализа.

<a id="fr-ml-cls-task-definition"></a>
### FR-ML-CLS-TASK-DEFINITION — Явная постановка

Эксперимент должен указать вход (весь объём или ROI), плоскости, стороны, классы и преобразование logits в решение. Сагиттальный эксперимент не подтверждает обученность фронтального выхода только потому, что модель содержит две головы.
Приёмка: [TC-ML-CLS-TASK-DEFINITION](#tc-ml-cls-task-definition).

<a id="tc-ml-cls-task-definition"></a>
### TC-ML-CLS-TASK-DEFINITION — проверка

- Предусловие: модель содержит несколько выходов, loss использует один.
- Действие: сверить протокол и отчёт.
- Результат: результат относится только к обученным и оценённым задачам.
- Покрытие: [artifact regressions](../../../../tests/test_cv_artifacts.py) проверяют sagittal-only checkpoint, architecture reconstruction, threshold/decision contract и отказ несовместимой модели; это технические проверки, не качество.

<a id="fr-ml-cls-group-independence"></a>
### FR-ML-CLS-GROUP-INDEPENDENCE — Групповая независимость

Все сравниваемые классификаторы должны использовать зафиксированные группы пациентов. При ROI от обученного детектора необходимо учитывать его учебные группы. Отсутствие пересечения сторон или серий не заменяет отсутствие пересечения пациентов.
Приёмка: [TC-ML-CLS-GROUP-INDEPENDENCE](#tc-ml-cls-group-independence).

<a id="tc-ml-cls-group-independence"></a>
### TC-ML-CLS-GROUP-INDEPENDENCE — проверка

- Предусловие: есть несколько моделей и сведения о detector train.
- Действие: проверить принадлежность всех звеньев к группам.
- Результат: сравнение использует одинаковые группы; утечка через детектор отсутствует или явно ограничивает вывод.
- Покрытие: Тест StratifiedGroupKFold существует; состав detector train требует аудита. [GAP-ML-DETECTOR-LEAKAGE](../../open-questions/README.md#gap-ml-detector-leakage).

<a id="fr-ml-cls-model-selection"></a>
### FR-ML-CLS-MODEL-SELECTION — Отделение подбора от итоговой оценки

Выбор эпохи, гиперпараметров и порога должен быть отделён от данных, на которых заявляется итоговое независимое качество. CV с выбором эпохи по validation того же фолда публикуется как оценка разработки, а не final test.
Приёмка: [TC-ML-CLS-MODEL-SELECTION](#tc-ml-cls-model-selection).

<a id="tc-ml-cls-model-selection"></a>
### TC-ML-CLS-MODEL-SELECTION — проверка

- Предусловие: известны данные выбора checkpoint и вычисления метрик.
- Действие: проверить протокол и формулировку вывода.
- Результат: источник каждого решения указан; независимый результат заявлен только при отделённой оценке.
- Покрытие: CV явно сообщает development_cv и model_selection_source=validation; regression — [test_sagittal_binary_cv.py](../../../../tests/test_sagittal_binary_cv.py). Binary выбирает эпоху по val AUC; shared multiclass — по val macro F1. [GAP-ML-VALIDATION-SELECTION](../../open-questions/README.md#gap-ml-validation-selection).

<a id="fr-ml-cls-threshold-calibration"></a>
### FR-ML-CLS-THRESHOLD-CALIBRATION — Калибровка порога

Порог должен иметь зафиксированный метод и источник калибровки без обращения к итоговым меткам. Для воспроизводимого порога оценочные преобразования входа должны быть детерминированными; train augmentation не считается отключённой вызовом model.eval().
Приёмка: [TC-ML-CLS-THRESHOLD-CALIBRATION](#tc-ml-cls-threshold-calibration).

<a id="tc-ml-cls-threshold-calibration"></a>
### TC-ML-CLS-THRESHOLD-CALIBRATION — проверка

- Предусловие: заданы calibration records и обучающий dataset с augmentation.
- Действие: повторить получение calibration predictions в оценочном режиме.
- Результат: входы воспроизводимы; порог и его происхождение сохранены; итоговая выборка не участвовала.
- Покрытие: CV использует отдельный ordered unaugmented loader training records; [test_sagittal_binary_cv.py](../../../../tests/test_sagittal_binary_cv.py) проверяет повторимость predictions. Источник training_unaugmented записан в отчёт; независимая calibration cohort не подразумевается.

<a id="fr-ml-cls-classification-report"></a>
### FR-ML-CLS-CLASSIFICATION-REPORT — Полный отчёт классификации

Отчёт должен различать плоскости, support классов, базовое сравнение и ошибки. Для бинарной задачи нужны AUC, accuracy, balanced accuracy, sensitivity/specificity, F1 с явным положительным классом и confusion matrix; для трёх классов — macro-F1, balanced accuracy и recall каждого класса. Коллапс в большинство нельзя скрывать средней accuracy.
Приёмка: [TC-ML-CLS-CLASSIFICATION-REPORT](#tc-ml-cls-classification-report).

<a id="tc-ml-cls-classification-report"></a>
### TC-ML-CLS-CLASSIFICATION-REPORT — проверка

- Предусловие: есть несбалансированные labels и константный классификатор.
- Действие: сформировать отчёт рядом с baseline.
- Результат: одноклассовые случаи и нераспознанные классы видны; метрики имеют определённую семантику.
- Покрытие: [Binary metrics tests](../../../../tests/test_binary_metrics.py) и [CV regressions](../../../../tests/test_sagittal_binary_cv.py) проверяют explicit positive-F1, support, recalls и train-majority baseline; [Shared multiclass](../../../../tests/test_multiclass_research.py) проверяет per-class precision/recall/F1/support, macro F1, balanced accuracy и train-majority baseline; исторический whole-volume trainer остаётся отдельным путём. [GAP-ML-INCOMPLETE-CLASS-REPORT](../../open-questions/README.md#gap-ml-incomplete-class-report).

<a id="fr-ml-cls-scientific-conclusion"></a>
### FR-ML-CLS-SCIENTIFIC-CONCLUSION — Сила научного вывода

Сводка должна показывать все запланированные фолды, способ агрегации, неопределённость и ограничения. Стандартное отклонение фолдов не называется доверительным интервалом. Достижение численной цели возможно только после выбора метрики, протокола и порога владельцем исследования.
Приёмка: [TC-ML-CLS-SCIENTIFIC-CONCLUSION](#tc-ml-cls-scientific-conclusion).

<a id="tc-ml-cls-scientific-conclusion"></a>
### TC-ML-CLS-SCIENTIFIC-CONCLUSION — проверка

- Предусловие: есть неполный прогон, среднее/std и заявленная цель.
- Действие: проверить итоговый вывод.
- Результат: неполнота отмечена, std назван корректно, несогласованная цель остаётся открытой.
- Покрытие: [artifact regressions](../../../../tests/test_cv_artifacts.py) проверяют private predictions→fold metrics→summary, protected run directory и failed snapshot с сохранённым completed fold. JSON содержит completion и population std; целевой критерий не утверждён. [OQ-ML-QUALITY-CRITERION](../../open-questions/README.md#oq-ml-quality-criterion).

## Версионированная конфигурация исследовательского запуска

Основной локальный runner и тонкий notebook используют один приватный JSON
schema_version=1 с mode=binary либо multiclass, явными input_path/dataset_root/output_dir и
параметрами существующей CV. Относительные пути разрешаются относительно config
file; outputs отделены от входов, включая ancestor/descendant containment.
Unknown keys/types, duplicate JSON, неверные явно заданные environment paths
и неоднозначное auto-discovery блокируют зависимый этап без раскрытия values.
Preflight проверяет canonical intake, paired ROI provenance и пригодность fold
групп до создания модели/обучения. Отчёт готовности содержит только агрегаты;
техническая готовность не подтверждает качество, применимость реальных меток
или detector independence. Облачный запуск не входит в этот локальный runner.
Приёмка: [TC-ML-READINESS-PREFLIGHT](../../features/dataset-readiness/README.md#tc-ml-readiness-preflight),
[TC-ML-READINESS-RUN](../../features/dataset-readiness/README.md#tc-ml-readiness-run).


Реализация: [единый runner](../../../../tools/run_research.py),
[strict path helper](../../../../training/utils/datasphere_env.py),
[config/preflight/CLI/notebook tests](../../../../tests/test_run_research.py) и
[path regressions](../../../../tests/test_datasphere_env.py).
Операторский config и команды: [runbook](../../../../google_colab/README.md).
Проверка выполнена на synthetic CPU inputs, без живого DataSphere/GPU прогона.


## Сагиттальный multiclass на per-side ROI

Тот же research config принимает mode=multiclass, отображая исходные
sagittal codes 1/2/3 в classes 0/1/2 central/anterior/posterior. Frontal 4/5/6
остаётся отдельным optional intake contract; отсутствие метки не становится
normal, неподготовленные frontal heads не выдаются. Binary defaults сохраняются.
Multiclass использует три logits и CE, default augmentation none; пространственные
flips/rotations без доказанного axis/remap блокируются. Patient folds требуют все
три класса в train и validation. Model selection — validation macro F1, ties
оставляют раннюю эпоху; development_cv не считается test, Youden не применяется.
Отчёт включает per-class precision/recall/F1/support, confusion matrix, macro F1,
balanced accuracy и train-derived majority baseline (ties lowest class index).
Checkpoint и private rows явно задают mode/classes/argmax tie rule и preprocessing;
CPU reload должен воспроизвести predictions и recomputation агрегатов.
Приёмка: [TC-ML-READINESS-MODES](../../features/dataset-readiness/README.md#tc-ml-readiness-modes).


Покрытие multiclass: [actualCPU/config/checkpoint/metrics/privacy tests](../../../../tests/test_multiclass_research.py).
[Dataset](../../../../training/datasets/tmj_position_dataset.py), [ROI model/loader](../../../../models/tmj_binary_position_classifier.py)
и [CV loop](../../../../training/sagittal_binary_cv.py) общие с binary; отдельного trainer нет.
Synthetic CPU проверки не устанавливают качество или detector independence.
