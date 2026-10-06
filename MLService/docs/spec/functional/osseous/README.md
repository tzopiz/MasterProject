# Исследование костных изменений по TMJ-OD3D

Это отдельная исследовательская задача `tmj-osseous-author-roi-v1`.
Она использует опубликованную авторскую разметку TMJ-OD3D V2 и не определяет
сагиттальное или фронтальное положение сустава. Вход модели — авторские ROI;
оценка на них не подтверждает классификацию произвольного DICOM без локализации.
Реализация и фактические проверки связаны в [трассировке](../../traceability/README.md).

<a id="fr-ml-osseous-labels"></a>
## FR-ML-OSSEOUS-LABELS — Семантика источника

Источник: DOI [10.57760/sciencedb.37727](https://doi.org/10.57760/sciencedb.37727),
релиз V2. Codebook фиксируется по [README авторов](https://github.com/ZiTingW/TMJ-OD3D/blob/a4bce89c89b7175e330193f04a282887b5e3f8b0/README.md):

| Код | Авторское значение |
|---|---|
| 0 | Normal |
| 1 | Cortical erosion |
| 2 | Subchondral sclerosis |
| 3 | Subchondral cystic changes |
| 4 | Condylar flattening |
| 5 | Osteophyte |
| 6 | Other |

Норма — только явная метка `[0]`. Binary target: `[0]` → 0, непустое сочетание
кодов 1–6 → 1. Сочетания патологий сохраняются; следующий режим — шесть
независимых multilabel выходов, а не взаимоисключающие классы. Отсутствие метки
или файла аннотации исключает сторону с причиной. Некорректные коды, конфликт
0 с патологией и несогласованность CSV/JSON не превращаются в норму.

Приёмка: [TC-ML-OSSEOUS-LABELS](#tc-ml-osseous-labels).

<a id="tc-ml-osseous-labels"></a>
### TC-ML-OSSEOUS-LABELS — Проверка авторских меток

Проверки всех кодов, комбинаций, пропусков,
конфликтов, односторонних и покадровых аннотаций; фактический CSV сверяется с
фиксированным checksum. Значения чужого источника не попадают в position intake.

<a id="fr-ml-osseous-preparation"></a>
## FR-ML-OSSEOUS-PREPARATION — Подготовка авторских ROI

Интервалы JSON включают обе границы и относятся к DICOM `InstanceNumber`.
Координаты `(x,y)` — столбец и строка; прямоугольник строится из диагональных
точек. Кандидатами служат только реально опубликованные срезы в интервале.
Пропуски не заполняются интерполяцией. Нерегулярный шаг не объявляется регулярным
трёхмерным объёмом: модель получает bag отдельных двумерных срезов.

Файл `image_path` может отсутствовать в релизе; это учитывается в provenance,
если существуют подходящие реальные срезы и подтверждены размеры/границы ROI.
Если файл присутствует, его InstanceNumber обязан попадать в заданный интервал.
Непустой набор подходящих срезов обязателен для каждого прямоугольника.
Пустой JSON без shapes исключает только эту сторону с причиной empty_annotation.

Профиль подготовки: native single-frame monochrome DICOM, одна серия и
согласованные размеры, orientation и pixel spacing; строгие ограничения объёма
до декодирования. Значения rescale проверяются, нормализация каждого исходного
среза — percentiles 1–99. До 16 выборок на сторону, bilinear resize до 96×96,
float16 `[K,1,H,W]`. При коротком интервале реально имеющиеся срезы могут
повторяться; новые экземпляры срезов не выдумываются.

Приватный индекс связывает patient/side, метки, Instances и fingerprints
исходных байтов, rescaled pixels, аннотаций, crop и prepared pixels. HTTP Range
и TAR проверяются строго; завершённость полного релиза требует EOF и охвата
всех строк CSV. Частичный прогон сохраняет checkpoint и сам по себе не готов
к cloud. Явно выбранная archive-prefix когорта проходит отдельную bounded policy:
точные count/offset и SHA256 source state, все выбранные IDs входят в закреплённый
CSV, непрерывные receipts заканчиваются на том же offset. Source complete=false
не изменяется; curated complete=true означает подготовку выбранной когорты.
Индекс сохраняет cohort.source_complete=false, release/selected counts и offset.
Это не случайная репрезентативная выборка полного релиза.
Обнаруженные противоречия источника рассматриваются отдельно. Исключение
непригодного пациента требует явной приватной review policy, привязанной к
checksum завершённого source state и точному failure receipt. Производный
индекс сохраняет исходные failure counts и provenance решения; ошибки не
стираются ради готовности. Неизвестная или непроверенная причина блокирует
курацию. См. [изменение 008](../../../../../docs/changes/008-od3d-curation/README.md).

Приёмка: [TC-ML-OSSEOUS-PREPARATION](#tc-ml-osseous-preparation).

<a id="tc-ml-osseous-preparation"></a>
### TC-ML-OSSEOUS-PREPARATION — Проверка подготовки

Отказы при повреждённых диапазонах, путях,
геометрии и checksum; безопасное resume без повторного добавления пациентов;
реальный полный прогон либо явно выбранная bounded cohort с проверенной
policy, отчётом exclusions/failures и consumer preflight.
Наблюдение нескольких кропов не доказывает анатомическую пригодность всей когорты.

<a id="fr-ml-osseous-evaluation"></a>
## FR-ML-OSSEOUS-EVALUATION — Фиксированная оценка

Обе стороны одного пациента остаются вместе. Deterministic patient split
70/15/15, seed 42, стратификация по наличию патологии хотя бы на одной стороне.
Пациенты технической отладки явно резервируются вне test. Membership и digest
сохраняются до обучения; повторный запуск проверяет тот же frozen split.
Точные дубликаты исходных/prepared pixels у разных patient identities запрещены.
Обе бинарные категории обязательны в каждом partition.

Baseline: shared 2D CNN с mean pooling срезов, BCEWithLogitsLoss, AdamW lr 0.001,
batch ≤4, ≤40 эпох, patience ≤8, без аугментации. Веса loss вычисляются только
по train. Выбор эпохи: validation AUROC для binary, macro AUPRC для multilabel;
пороги выбираются по validation. Test не участвует в настройке. Для редких
меток без train/validation class support сохраняются отдельные support masks,
их пересечение, причины ограничений и происхождение fallback порога 0.5.
Такие выходы не входят в aggregate AUROC/AUPRC/bootstrap; per-label метрики
равны null, диагностические probabilities/confusion сохраняются. Полное
отсутствие совместно поддерживаемой метки отклоняется уже в preflight.
Support masks, selection и calibration не используют test labels.

Отчёт содержит AUROC/AUPRC, confusion, sensitivity/specificity, support,
constant train-prevalence baseline и patient bootstrap CI (200 выборок).
Короткий технический запуск может отключать bootstrap; он не считается
подтверждением качества. Отрицательный результат исследования допустим.

Приёмка: [TC-ML-OSSEOUS-EVALUATION](#tc-ml-osseous-evaluation).

<a id="tc-ml-osseous-evaluation"></a>
### TC-ML-OSSEOUS-EVALUATION — Проверка оценки

CPU обучение обоих режимов, сохранение и
восстановление checkpoint, запрет patient/duplicate leakage, неизменность
выбранной эпохи и порогов при изменении test labels. Публичный отчёт содержит
агрегаты; predictions, membership, checkpoints и index остаются приватными.

<a id="fr-ml-osseous-architecture"></a>
## FR-ML-OSSEOUS-ARCHITECTURE — Явный экспериментальный вариант

Config `architecture` принимает только `global_mean` (default, прежняя модель)
или `spatial_head` (экспериментальный вариант; критерий TRAIN-only CV не выполнен).
Spatial наследует тот же conv trunk и mean pooling срезов, меняет финальную
AdaptiveAvgPool2d на 4×4 и head на Linear(256,32), ReLU, Linear(32,outputs).
Binary/multilabel имеют 1/6 outputs. Остальные параметры baseline неизменны.
Технический TRAIN-only fit probe мотивирует доступность для воспроизведения.
Patient CV не прошла критерий выбора; вариант не выбран для validation.

Metadata/config checkpoint и публичный report фиксируют имя; reload проверяет
согласованность metadata/config/bindings. Старые отсутствующие имена означают
`global_mean`. Только `spatial_head` добавляет имя в bindings, чтобы не менять
прежние immutable bundles. Неизвестное имя отклоняется безопасным
`unsupported_architecture`; конфликт — `checkpoint_architecture_mismatch`.
Cloud staging сохраняет выбор, consumer проверяет report/binding и digests.

Приёмка: [TC-ML-OSSEOUS-ARCHITECTURE](#tc-ml-osseous-architecture).

<a id="tc-ml-osseous-architecture"></a>
### TC-ML-OSSEOUS-ARCHITECTURE — Совместимость и восстановление

Оба spatial development режима работают без test NPZ и воспроизводят checkpoint
predictions. Порядок срезов не меняет logits; default и legacy checkpoint дают
точные прежние logits/bindings. Неизвестные/несогласованные имена отклоняются,
cloud staging сохраняет вариант и старые reports остаются допустимыми.
Связь: [изменение 012](../../../../../docs/changes/012-osseous-architecture/README.md).

<a id="fr-ml-osseous-development"></a>
## FR-ML-OSSEOUS-DEVELOPMENT — Эксперименты без test inference

`train(config, development=True)` / CLI `--develop` требуют существующий frozen
split. Test membership/declared provenance проверяются, но test NPZ не открываются
и их payload integrity не подтверждается. Train/validation crop checks остаются
строгими. Membership и исходный index digest не изменяются; binding дополнительно
содержит evaluation_scope=development. Выбор эпохи и порогов использует validation.

Отчёт обозначен development и содержит train/validation metrics; test metrics,
predictions и crop bindings отсутствуют. Validation используется при подборе,
поэтому её метрики и bootstrap не являются независимой оценкой качества.
История сохраняет среднюю weighted BCE по обучающим minibatches, validation
weighted BCE и std validation probabilities. При inference/reporting порядок
соответствует targets. Обычный holdout путь по-прежнему проверяет test payload.
Разработка не создаёт новые права на публикацию приватных данных/весов или
повторное платное обучение.

Приёмка: [TC-ML-OSSEOUS-DEVELOPMENT](#tc-ml-osseous-development).

<a id="tc-ml-osseous-development"></a>
### TC-ML-OSSEOUS-DEVELOPMENT — Проверка изоляции

Binary/multilabel development успешно завершаются при удалённых test crops,
сохраняют checkpoint/reload и только train/validation predictions/manifest.
Без frozen split либо с неверным digest выполнение отклоняется до создания
output. Обычный preflight с отсутствующими test crops отклоняется.
Связь: [изменение 011](../../../../../docs/changes/011-osseous-development/README.md).

<a id="fr-ml-osseous-cloud"></a>
## FR-ML-OSSEOUS-CLOUD — Подтверждение конкретного запуска

Подготовка bundle выполняется локально и не запускает удалённые задания.
Она требует завершённой подготовки выбранной когорты без необработанных
failures и с сохранённым scope/provenance, повторяет
consumer preflight после staging и фиксирует bytes/hash кода, данных, config,
split, codebook и mapping. Payload ≤5 GiB. Confirm принимает SHA256 плана;
повторная отправка после неизвестного исхода запрещена до reconciliation.

Requirements-file для официального CLI содержит только package specifiers и поддерживаемые pip flags, без комментариев. Plain системный контейнер `system-python-3-10`: bootstrap внутри manifest проверяет Python 3.10, создаёт isolated venv без system-site-packages, устанавливает закреплённые пакеты через pip --python, проверяет imports/CUDA и только затем запускает worker. PYTHONPATH/PYTHONHOME очищаются; env.python и дополнительный local-module upload не объявляются.

План указывает project/profile, GPU, Python/requirements, стоимость на дату,
время обучения и исключённые расходы. Watchdog ограничивает процесс обучения
четырьмя часами; это не гарантия предельной суммы счёта, подготовка среды и
хранилище оплачиваются отдельно. Платный запуск требует подтверждения плана.
Provider SUCCESS не заменяет проверку training-status, completion и всех
artifact digests против ожидаемых bindings конкретного bundle.

Приёмка: [TC-ML-OSSEOUS-CLOUD](#tc-ml-osseous-cloud).

<a id="tc-ml-osseous-cloud"></a>
### TC-ML-OSSEOUS-CLOUD — Проверка cloud-пакета

Подготовка без provider IO, неизменяемый staging,
изменённые bytes и несовпадающие result bindings отклоняются; timeout завершает
свою группу процессов. Финальная готовность требует настоящего индекса выбранной когорты, frozen
split, preflight и готового bundle для доступного проекта.
Проверка на CPU не подтверждает уже выполненный GPU запуск.

Порядок запуска и фактический статус: [протокол](../../../../experiments/tmj_od3d_feasibility_20261005/README.md),
[изменение 003](../../../../../docs/changes/003-tmj-od3d-pilot/README.md).
