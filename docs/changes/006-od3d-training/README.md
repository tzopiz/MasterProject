# 006 — Воспроизводимое обучение костных изменений

Статус: in-progress. Основание: [005](../005-od3d-preparation/README.md).

## План и критерии приёмки

1. Проверять фактические crop bytes, семантику меток и provenance отдельной
   задачи tmj-osseous-author-roi-v1; binary и multilabel выходы не смешиваются
   с классификацией положения сустава.
2. Фиксировать patient split 70/15/15, seed 42. Обе стороны остаются вместе;
   development-only пациенты исключаются из test. Отказывать при дубликатах
   исходных/prepared pixels, неверных digests, утечке между partitions и
   недостатке train/validation class support для multilabel selection.
3. Реализовать shared 2D CNN с mean pooling, BCE loss с train weights,
   AdamW lr 0.001, batch ≤4, ≤40 эпох, patience ≤8. Выбор эпохи и порогов
   только по validation. Frozen membership сохраняется до обучения.
4. Сравнивать с constant train-prevalence baseline; AUROC/AUPRC, confusion,
   sensitivity/specificity и patient bootstrap 200 draws. Не выдавать rare
   labels без train/validation support за проверенный результат.
5. Приватно сохранять checkpoint, predictions и membership; публично только
   агрегаты. Проверять восстановление checkpoint и completion bindings.
6. Реальный полный consumer preflight обоих режимов и CPU smoke обязательны
   до готовности DataSphere bundle. Улучшение метрик оценивается после обучения.

## Этап preflight

Реализованы шаги 1–2: consumer проверяет bytes/shape/dtype/range, source and
prepared pixel identity, task/codebook, membership/digest и train/validation
support. Preflight не создаёт output и не обучает модель.

Шаги 3–6 остаются открытыми в этом review unit. Частичный индекс и синтетические
проверки не подтверждают готовность всей реальной когорты.

Проверки этого этапа: `python -m pytest
MLService/tests/test_tmj_osseous_research.py -q -p no:cacheprovider` в
cloud-pins-venv — **16 passed in 0.75s**. Проверки F/I без cache и
`python3 scripts/check_specs.py` — PASS. Обучение на этом этапе не запускалось.
