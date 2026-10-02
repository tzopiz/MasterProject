# Пробелы и открытые решения MLService

GAP — доказанное статической сверкой расхождение с требуемой семантикой.
OQ — решение владельца исследования; это не найденный дефект реализации.
Синтетические проверки подтверждают только технический путь. Реальная когорта
и медицинское качество ими не проверяются. Закрытие записи требует изменения
и указанной приёмки, а не переименования ограничения.

<a id="gap-ml-patient-identity"></a>
### GAP-ML-PATIENT-IDENTITY — Идентичность не подтверждена
Основной canonical путь требует source_id/patient_id/study_id и проверяет явную применимость разметки; обе стороны и повторные исследования группируются по источнику и пациенту. TC-ML-DATA-PATIENT-GROUP проверяется на синтетическом mapping. Имя больше не служит основным ключом; name join сохранён только как explicit legacy. Достоверность supplied patient mapping реальной когорты остаётся входным условием, а не выводится из синтетики.

<a id="gap-ml-label-join-validation"></a>
### GAP-ML-LABEL-JOIN-VALIDATION — Неполный контроль связывания
Закрыто для canonical intake: неизвестные поля, дубликаты, ошибочные типы/коды, отсутствие метки и неоднозначная применимость вызывают структурированный отказ. TC-ML-DATA-LABEL-JOIN и TC-ML-DATA-LABEL-SEMANTICS покрыты синтетическими проверками. Исторический name join остаётся ограниченным legacy путём.

<a id="gap-ml-derived-data-provenance"></a>
### GAP-ML-DERIVED-DATA-PROVENANCE — Производные не имеют полного паспорта
Основной paired writer исправлен: строгие sidecar содержат source/detector hashes, версии preprocessing, сторону, checksum, shape и transform; кэш принимается только после полной проверки. TC-ML-DATA-DATA-PROVENANCE проверяет изменения входов/настроек/весов. Crop-only режим явно сообщает об отсутствии повторной проверки raw source. Исторические кропы без паспортов не приобретают подтверждённое происхождение автоматически.

<a id="gap-ml-pipeline-geometry"></a>
### GAP-ML-PIPELINE-GEOMETRY — Геометрия между путями не согласована
Основной paired research путь сохраняет объявленный InstanceNumber порядок, проверяет поддерживаемую геометрию, правильный requested origin/padding и physical affine либо явный voxel-space. TC-ML-LOC-COORDINATES и TC-ML-LOC-ROI-GEOMETRY покрыты синтетическими landmarks/roundtrip, включая float32 NIfTI сериализацию. Поддерживаемый ограниченный профиль указан в локализации. Другие исторические preprocessing-пути не считаются эквивалентными; качество/стороны реального детектора требуют проверки на данных.

<a id="gap-ml-detector-leakage"></a>
### GAP-ML-DETECTOR-LEAKAGE — Утечка через локализацию не исключена
Detector split формируется по сериям. Patient-grouped classifier split не
доказывает независимость от detector train. Закрытие: TC-ML-LOC-LOCALIZATION-EVALUATION, TC-ML-CLS-GROUP-INDEPENDENCE.

<a id="gap-ml-validation-selection"></a>
### GAP-ML-VALIDATION-SELECTION — Validation участвует в выборе эпохи
Выбор checkpoint по validation сохранён как явно обозначенный development_cv; отчёт указывает model_selection_source=validation и independent_test=false. Это удовлетворяет TC-ML-CLS-MODEL-SELECTION для оценки разработки. Независимая итоговая оценка требует отдельно подготовленной когорты/протокола; текущие CV числа ею не являются.

<a id="gap-ml-augmented-calibration"></a>
### GAP-ML-AUGMENTED-CALIBRATION — Калибровка использует аугментированный train
Закрыто для основного CV: порог определяется по отдельному неаугментированному ordered train loader; finite Youden threshold и правило >= сохраняются явно. TC-ML-CLS-THRESHOLD-CALIBRATION проверяется на синтетике. Исторический holdout trainer не становится рекомендованным путём автоматически.

<a id="gap-ml-incomplete-class-report"></a>
### GAP-ML-INCOMPLETE-CLASS-REPORT — Отчёт может скрывать коллапс
Закрыто для общего ROI research runner: multiclass выбирает checkpoint по validation macro F1, публикует precision/recall/F1/support каждого из трёх классов, confusion matrix, balanced accuracy и train-majority baseline. Binary отчёт явно называет positive=1 и сохраняет исторический f1_minority только как legacy alias. TC-ML-CLS-CLASSIFICATION-REPORT покрыт реальным коротким train/save/reload и пересчётом метрик из приватных строк. Исторический whole-volume trainer не является новым multiclass путём.

<a id="gap-ml-segmentation-preprocessing"></a>
### GAP-ML-SEGMENTATION-PREPROCESSING — Preprocessing сегментации расходится
2D dataset использует HU clip, evaluator — slice/max и resize.
Закрытие: TC-ML-SEG-EVALUATION-PREPROCESSING; результат нельзя автоматически считать сопоставимым.

<a id="gap-ml-segmentation-split"></a>
### GAP-ML-SEGMENTATION-SPLIT — Split масок не подтверждает независимость
Разбиения volumes/crops не задают patient groups; малый 3D набор может иметь
train=val. Закрытие: TC-ML-SEG-MASK-EVALUATION на зафиксированных группах.

<a id="gap-ml-cv-artifacts"></a>
### GAP-ML-CV-ARTIFACTS — CV-артефакты неполны
Закрыто для основного CV: отдельный защищённый каталог содержит report, checkpoint каждого fold, приватные predictions/membership и replay, fingerprints кода/входов/runtime. Повторный каталог отвергается, failed/partial сохраняет выполненную часть. TC-ML-NFR-EXPERIMENT-REPRODUCIBILITY проверяется настоящим коротким CPU train/save/reload и пересчётом метрик из приватных строк. Исторические runs без артефактов остаются невоспроизводимыми задним числом.

<a id="gap-ml-upload-validation"></a>
### GAP-ML-UPLOAD-VALIDATION — Upload trust boundary не защищена полностью
Filename соединяется с temp_dir напрямую; одинаковые имена перезаписываются,
ограничения размеров/числа файлов и геометрии не заданы. Закрытие: TC-ML-NFR-UPLOAD-BOUNDARY.

<a id="gap-ml-result-validation"></a>
### GAP-ML-RESULT-VALIDATION — DTO не валидирует полную геометрию
Lists не ограничены длинами/конечностью; согласованность bbox/shape не проверяется.
Закрытие: TC-ML-INT-RESULT-GEOMETRY, без заявления о выполненном wire-тесте.

<a id="oq-ml-repeated-study-label"></a>
### OQ-ML-REPEATED-STUDY-LABEL — Применимость повторной разметки
Canonical intake требует отдельную ссылку исследования на запись разметки, совпадающий patient_id и label_applicability=confirmed. Автоматического переноса метки по имени/номеру нет.

Ответ: правило реализовано; подтверждение применимости конкретной разметки относится к подготовке реального входного файла.

<a id="oq-ml-quality-criterion"></a>
### OQ-ML-QUALITY-CRITERION — Критерий целевого качества
Выбрать основную бинарную метрику и порог; «90+%» не определяет их однозначно.
Численный multiclass критерий также открыт. Предложение balanced accuracy
не является утверждённой приёмкой; исторические наблюдения её не заменяют.

Ответ: ожидается решение владельца.

<a id="oq-ml-final-evaluation-protocol"></a>
### OQ-ML-FINAL-EVALUATION-PROTOCOL — Итоговый протокол и сегментация
Определить доступную независимую когорту/схему вложенной оценки, необходимость
расширения данных и критерий сегментации. До решения численные гарантии не заданы.

Ответ: ожидается решение владельца.

<a id="oq-ml-input-resource-limits"></a>
### OQ-ML-INPUT-RESOURCE-LIMITS — Эксплуатационные ограничения входа
Согласовать лимиты ресурсов, поддерживаемую DICOM-геометрию/расширения, timeout
и политику ошибок в общей интеграции. Локальный прототип не задаёт SLA.

Ответ: ожидается решение владельца.
