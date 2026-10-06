# 017 — Исследовательская оценка шести osseous признаков

Статус: open

## Проблема и целевое изменение

После binary baseline нужно проверить multilabel capability. Использовать
существующие TRAIN mean16 features и API для6outputs, сохранённые combinations
и patient folds. Binary best и public API не меняются. Оценка редких классов
должна явно показывать support и null, а не скрываться за macro score.
Связь FR-ML-OSSEOUS-LABELS / FR-ML-OSSEOUS-FEATURES /
FR-ML-OSSEOUS-EVALUATION / FR-ML-OSSEOUS-DEVELOPMENT.
[Протокол](../../../MLService/experiments/osseous_multilabel_cv_20261006/README.md).

## Приёмка

- Fixed hashes и target OR parity; codes1–6/multi-hot сохраняют combinations.
- Exact5outer/3inner patient folds, TRAIN-only cache; no originalval/test payload.
- Fresh scaler/head, sharedC выбран внутриouterfit по fixed support mask macroAP.
- Six outputs в code order, fit-only constant fallback, явные masks/null/reasons.
- Per-label patient support, baseline на тех же eligible rows, bootstrap fixed masks.
-≤95bundle fits,180s cap, exact checkpoints; приватные artifacts600/700.
- Independent review и фактические aggregate results, registry/specs checks,
  отдельный bounded PR. Отрицательный результат не меняет binary best.

## План

1. Preregister/peer review; private worker без изменения public API.
2. CPU execution, verify artifacts/targets/support/checkpoints/metrics.
3. Обновить protocol/registry результатами, independent result review и PR.

## Результаты

2026-10-06: preliminary TRAIN label-count audit, протокол заполнен.
Fits/predictions/quality и publication acceptance пока не выполнены.
