# 009 — Фиксированная когорта без дальнейшего добора

Статус: done. Пользователь 2026-10-05 остановил добор публичных данных,
сохранив цель: всё готово к запуску обучения. За время остановки получены
148 исследований и 289 пригодных side records; source failures: два пациента
с annotation_side_conflict. Исходный релиз по-прежнему содержит 1043 пациента.

Связанные требования: [FR-ML-OSSEOUS-PREPARATION](../../../MLService/docs/spec/functional/osseous/README.md#fr-ml-osseous-preparation),
[FR-ML-OSSEOUS-EVALUATION](../../../MLService/docs/spec/functional/osseous/README.md#fr-ml-osseous-evaluation),
[FR-ML-OSSEOUS-CLOUD](../../../MLService/docs/spec/functional/osseous/README.md#fr-ml-osseous-cloud).

## Целевая разница и приёмка

Default full-release curation продолжает требовать complete=true, EOF и
1043 metadata coverage. Для явно выбранной bounded cohort policy задаёт
kind=archive-prefix, source_patient_count и source_end_offset. Source checkpoint
не меняется: complete=false; никакого ложного EOF или 1043 coverage.
Policy по-прежнему связана с точным SHA256 source state и всеми reviewed failures.

Все выбранные source IDs должны быть уникальны и присутствовать в закреплённом
CSV; количество metadata rows равно release count. Receipt count, непрерывность
ranges, last receipt end и checkpoint offset совпадают с явным bounded policy.
Число выбранных пациентов положительно и меньше release count; offset положителен
и меньше ARCHIVE_BYTES. Несовпадение count/hash/offset, непроверенные failures,
неизвестный source ID или live source lock блокируют подготовку.

Curated complete=true означает завершённую подготовку выбранной когорты.
Индекс явно содержит cohort kind/source_complete=false, release/selected counts
и source offset. Это archive prefix, не случайная репрезентативная выборка.
Нельзя использовать результат для утверждений о всей популяции.

## План

1. Добавить opt-in bounded policy в существующую curation; default gates сохранить.
2. Behavioral tests: bounded success; неправильные count/offset/type/coverage,
   неполная receipt coverage и отсутствие opt-in отклоняются; source untouched.
3. Приватно проверить каждый source failure и записать policy; выполнить курацию
   сохранённого checkpoint без новых загрузок и без копирования raw DICOM.
4. Real-data preflight binary/multilabel; frozen patient split, development-only
   вне test. Короткий CPU smoke проверяет исполнимость, не качество модели.
5. Собрать два immutable DataSphere bundles, проверить их CLI schema, bindings,
   frozen membership, доступность проекта и актуальные ресурсы/стоимость.
6. Независимое ревью, checks/spec updates/PR. Платный запуск не выполняется
   до подтверждения конкретного плана. Готовность требует всех фактических gates.

## Результаты

Загрузка и supervisor остановлены; живых downloader процессов нет. Согласованный
checkpoint/state/index и 289 crops скопированы в приватное игнорируемое хранилище
проекта: SHA256 всех 292 файлов проверены, около 62.3 MB. Raw DICOM не перенесены.


Добавлен opt-in `policy.cohort` в существующий `curate_tmj_od3d.py`:
только `archive-prefix` с точными integer count/offset, complete=false и полным
закреплённым metadata release. Default сохраняет complete=true/1043/EOF.
Проверяются subset/уникальность, receipts, reviewed exclusions, SHA256 policy
binding, исходные файлы и mutation gates. Curated index/report содержат только
агрегированный cohort provenance; source state и исходные имена не публикуются.
Production diff: +21/−5 строк; behavioral tests отдельно: +104/−0 строк.

Проверки (synthetic fixtures, без сети/GPU или новых загрузок):
- RED: `MLService/docs/ai/tasks/GH-85/cloud-pins-venv/bin/python -m pytest MLService/tests/test_curate_tmj_od3d.py -q --tb=short`
  — 26 expected failures, 31 passed; success prefix ещё отклонялся source_incomplete.
- GREEN тем же targeted command — 57 passed.
- `MLService/docs/ai/tasks/GH-85/cloud-pins-venv/bin/python -m pytest MLService/tests -q --tb=short`
  — 617 passed за 41.39 s; затем уточнён synthetic receipt-gap fixture до
  однобайтового разрыва (проверяет непрерывность при корректном lo<hi).
Эта проверка подтверждает код курации, но не заменяет real-data preflight и
не утверждает завершённость общего плана 009.

После уточнения receipt-gap: targeted suite — 57 passed за 0.80 s;
`python3 scripts/check_specs.py` — OK; scoped `git diff --check` — OK.

## Фактическая приёмка выбранного запуска

План 1–5 закрыт. Реальная curation сохранённого checkpoint прошла без сети:
148 source cases → 146 retained patients / 289 sides, два source failures
annotation_side_conflict явно исключены. Для каждого receipt проверены range,
нулевое side_count и единственный pinned validator code path: несовпадение
author shape side с ожидаемой стороной L/R файла. Первый исходный JSON ранее
осмотрен отдельно; повторный осмотр второго raw JSON не заявляется.
Никакие стороны не исправлены предположениями, original failures сохранены.

Source complete=false и исходные bytes сохранены; curated cohort указывает
source_complete=false, 148 selected / 1043 release и точный checkpoint offset.
Real consumer preflight binary/multilabel: PASS; partition 102/22/22 пациентов
и 204/43/42 стороны. Frozen membership/digest сохранены до обучения;
development-only три пациента вне test. Все шесть меток поддержаны train/val.

Два immutable bundle: 61910529 bytes binary и 61910533 bytes multilabel;
staged preflight, manifest/job hash, frozen membership и установленный
DataSphere CLI parse_config/validate_paths — PASS. Два development ROI из
train staged payload прошли forward/BCE/backward/AdamW step обоих режимов
на Torch 2.6.0 CPU; test не оценивался.

Read-only DataSphere API на 2026-10-05 вернул 200 для проекта и restrictions
проекта/сообщества; ALLOW_JOBS и ALLOW_SPEC_GT_4_1 true. План: gt4.1/T4,
Python 3.12, Torch 2.6.0+cu118 и явный CUDA 11.8 Docker image; до 4 часов
обучения. [Официальная ставка](https://yandex.cloud/ru/docs/datasphere/pricing)
168.48 RUB/hour, training-window estimate 673.92 RUB на режим, setup/storage/
egress отдельно; hard money cap не обещается. Paid submission не выполнен.

Независимое read-only ревью bounded curation не нашло actionable bugs:
default full-release gates сохранены, explicit opt-in и source hash/count/
offset проверяются, fake EOF не создаётся, provenance входит в staged index.
617 ML tests, финальные 57 curation tests, F/I lint и спецификации проходят.
Изменения подготовлены отдельным review unit; приватные планы, membership,
policy, data и credentials не включаются в Git. Остаётся подтвердить запуск.

Дополнительно сверена реализация [AdaptiveAvgPool2d(1) в PyTorch 2.6](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/AdaptiveAveragePooling.cpp):
для используемого non-quantized выхода 1×1 она вычисляет mean по пространственным
осям. Worker передаёт CUBLAS_WORKSPACE_CONFIG до старта training subprocess.
Это проверка совместимости кода/конфигурации, не выполненный CUDA запуск.
