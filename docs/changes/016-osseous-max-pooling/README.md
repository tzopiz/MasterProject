# 016 — Проверка max pooling для frozen osseous features

Статус: done

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

2026-10-06: private worker414строк independently accepted SHA
`64a9b99489ea8b48c1a3d35914b431566cb87677c5e86903a139ea0b1b5b78cf`.
CPU run completed exit0:204TRAIN bags,90inner/5outer fits,80,41с+1,16с;
mean parity0,0,15completion hashes и5exactcheckpointreplays проверены root.
MeanCV AUROC0,640263, delta−0,047668, wins2/5; criterion failed, retain_nested_mean16.
Original validation/test не открывались; source/API/defaults не изменены.
Все три пункта выполнены: worker/protocol/result/docs independently reviewed,
результаты/registry обновлены и [PR115](https://github.com/tzopiz/MasterProject/pull/115) опубликован.
Aggregate data и hashes — в протоколе; приватные artifacts не в git.

`python3 scripts/check_specs.py`, `--self-test`, `git diff --check` →passed.
