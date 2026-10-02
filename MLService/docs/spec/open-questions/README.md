# Пробелы и открытые решения MLService

GAP — доказанное статической сверкой расхождение с требуемой семантикой.
OQ — решение владельца исследования; это не найденный дефект реализации.
Проверка runtime и реальной когорты здесь не выполнялась. Закрытие записи
требует изменения и указанной приёмки, а не переименования ограничения.

<a id="gap-ml-patient-identity"></a>
### GAP-ML-PATIENT-IDENTITY — Идентичность не подтверждена
Join/grouping используют stripped name; организатор назначает subject_id каждой
серии. Одинаковые/разные написания могут нарушить группы. Закрытие: TC-ML-DATA-PATIENT-GROUP
на проверенном patient mapping, без раскрытия имён в публичном отчёте.

<a id="gap-ml-label-join-validation"></a>
### GAP-ML-LABEL-JOIN-VALIDATION — Неполный контроль связывания
Повторный name_raw молча перезаписывает метки; unmatched studies пропускаются.
Полная схема/legend и список исключений отсутствуют. Закрытие: TC-ML-DATA-LABEL-JOIN, TC-ML-DATA-LABEL-SEMANTICS.

<a id="gap-ml-derived-data-provenance"></a>
### GAP-ML-DERIVED-DATA-PROVENANCE — Производные не имеют полного паспорта
Crop metadata не обеспечивает checksums/версию preprocessing; skip-existing
проверяет наличие двух файлов. Закрытие: TC-ML-DATA-DATA-PROVENANCE на двух версиях входов.

<a id="gap-ml-pipeline-geometry"></a>
### GAP-ML-PIPELINE-GEOMETRY — Геометрия между путями не согласована
Порядок срезов и preprocessing различаются; orientation не проверяется,
crop affine единичный. Закрытие: TC-ML-LOC-COORDINATES, TC-ML-LOC-ROI-GEOMETRY с известной геометрией.

<a id="gap-ml-detector-leakage"></a>
### GAP-ML-DETECTOR-LEAKAGE — Утечка через локализацию не исключена
Detector split формируется по сериям. Patient-grouped classifier split не
доказывает независимость от detector train. Закрытие: TC-ML-LOC-LOCALIZATION-EVALUATION, TC-ML-CLS-GROUP-INDEPENDENCE.

<a id="gap-ml-validation-selection"></a>
### GAP-ML-VALIDATION-SELECTION — Validation участвует в выборе эпохи
CV выбирает checkpoint по val AUC того же фолда, отдельного test нет.
Закрытие: TC-ML-CLS-MODEL-SELECTION; текущие числа остаются оценкой разработки.

<a id="gap-ml-augmented-calibration"></a>
### GAP-ML-AUGMENTED-CALIBRATION — Калибровка использует аугментированный train
CV собирает train predictions тем же loader с augmentation. Старый holdout
trainer калибрует на val. Закрытие: TC-ML-CLS-THRESHOLD-CALIBRATION с отделённой калибровкой.

<a id="gap-ml-incomplete-class-report"></a>
### GAP-ML-INCOMPLETE-CLASS-REPORT — Отчёт может скрывать коллапс
Multiclass trainer выбирает mean accuracy, без требуемых per-class метрик
и baseline; binary f1_minority означает positive=1. Закрытие: TC-ML-CLS-CLASSIFICATION-REPORT.

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
JSON не хранит membership, individual predictions, лучший checkpoint и полный
fingerprint входов; повторный run может перезаписать отчёт. Закрытие: TC-ML-NFR-EXPERIMENT-REPRODUCIBILITY.

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
Владелец разметки должен определить, когда одна пациентская запись относится
к нескольким сериям и как отражать изменение состояния. До решения — ограничение.

Ответ: ожидается решение владельца.

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
