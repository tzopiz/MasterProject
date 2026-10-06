# 016 — Проверка max pooling для frozen osseous features

Статус: open

## Проблема и целевая разница

Layer4 fine-tuning не прошёл критерий улучшения. Проверить одну дешёвую гипотезу:
max вместо mean по16 frozen slice embeddings, с неизменной размерностью512
и nested logistic head. Public API/defaults не меняются; private research entrypoint
и результаты ведутся по [протоколу](../../../MLService/experiments/osseous_max_features_cv_20261006/README.md).
Связь FR-ML-OSSEOUS-FEATURES / FR-ML-OSSEOUS-DEVELOPMENT / FR-ML-OSSEOUS-EVALUATION.

## Приёмка

- До fit закрепить dataset/crop/weights/code/cache/fold hashes; только TRAIN204bags.
- Точные прежние outer/inner memberships; fit-only scaler/C selection; max metadata.
- Повторное mean extraction совпадает с pinned cache≤1e-6; checkpoint reload exact.
- No original validation/test inference, patient data/cache/weights остаются приватны.
- Criterion и resource caps из протокола соблюдены; review и результат сохранены,
  отрицательный результат допустим без объявления модели улучшенной.
- Реестр/README обновлены; specs/diff checks, bounded independent review и PR.

## План

1. Preregister; private worker и independent review до запуска.
2. CPU extraction/nested fits, проверить artifacts/metrics/decision.
3. Сохранить aggregate evidence, обновить registry и опубликовать reviewed PR.

## Результаты

2026-10-06: протокол заполнен; запуск и результаты отсутствуют.
