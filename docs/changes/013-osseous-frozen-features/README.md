# 013 — Reusable frozen osseous features

Статус: done

## Проблема и целевое изменение

Frozen ResNet18 + регуляризованный head проверены отдельными приватными research
runners; после clone этот новый путь ещё нельзя воспроизвести через repo API.
Nested TRAIN-only CV meanAUROC0,688, originalvalidation0,727,source1510,571;
это exploratory authorROI result, не DICOM localization или clinical acceptance.

Добавить небольшой offline feature/head module и focused behavioral tests.
Existing scratch baseline/default/cloud bundles не менять. PublicCLI и nestedCV
orchestration — следующая самостоятельная задача после принятия library.
Связь FR-ML-OSSEOUS-EVALUATION / FR-ML-OSSEOUS-DEVELOPMENT.

## Приёмка

- OfflineResNet18ImageNetV1 loading проверяет полный fixedweights SHA256,
  не скачивает weights автоматически, использует safe torchweights loading,
  frozen/eval/inference_mode и stable512feature output.
- Strict normalized16×1×96×96 bag; wholeROIresize224,RGBrepeat,ImageNetnormalization,
  mean16sliceaggregation. Nonfinite/shape/range отклоняются безопасными codes.
- Fit-onlyStandardScaler + balanced LogisticRegression с explicitC,seed42,
  max_iter2000; binary1 или multilabel6 output в codebook order.
- Unsupported oneclasslabels сохраняют trainprevalence constant и supportmask;
  отсутствие поддержки не объявляется normal или quality evidence.
- Predict finite probabilities, serializable caller-owned bundle; errors не
  раскрывают paths/values/IDs. ConvergenceWarning вызывает safe refusal.
- Synthetic tests проходят без downloads/privateweights/пациентских данных;
  проверяют numerical transforms/frozen extractor/probability order/support,
  fit-only scaling, inputs/error cases. Independent review/ruff/specs required.

## План

1. Создать failing synthetic behavioral checks, реализовать bounded library.
2. Проверить edgecases, поддерживаемые и unsupportedbinary/multilabel outputs.
3. Обновить permanent spec/API navigation и traceability после реализации.
4. Выполнить focused checks и independent review; отдельный bounded PR.

## Результаты

2026-10-06: все четыре пункта выполнены. Новые module/test —223/271 строк после repo formatter;
31 synthetic test проходит без private cohort/download. Independent review
закрыт на SHA256 module5f0f5dd0ed3d0f3f7605d3b1a7ec120bf96213e2a8b66ffa7455c8ddf5128f03,
tests a8bd28893e74534dbe9c464fb6e45d8e90b56ae1ae772a472f24fab7a9275b08.

Фактические проверки:
- `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MLService/docs/ai/tasks/GH-85/cloud-pins-venv/bin/python -m pytest -q -p no:cacheprovider MLService/tests/test_tmj_osseous_features.py` →31passed (peer0,98s).
- Ruff check module/tests →passed; `git diff --check` →passed.
- `python3 scripts/check_specs.py` и `--self-test` →passed.
- Root private hash/order-bound replay: один TRAIN crop →точное совпадение512
  features; fullTRAIN C0,01 head →точное совпадение43 сохранённых validation
  probabilities (maxdifference0),0,80s; test inference=false.
- Permanent FR/TC-ML-OSSEOUS-FEATURES, API navigation и traceability обновлены.

Synthetic проверки и replay подтверждают recipe compatibility. Public CLI и
orchestration остаются отдельной задачей; API не обещает clinical readiness.
Приватные research результаты доступны только агрегатами в
[реестре экспериментов](../../../MLService/experiments/README.md).

CI выявил только форматирование новых tests; все три Python3.10/3.11/3.12
pytest jobs и pinnedCPU smoke прошли. Formatter применён к новым module/tests,
AST до/после точно совпал;31 focused tests повторно passed. Новые SHA:
module1a14a2f9ea097d18f03e63f497b1f4e924a417116a6e636ca7d0f9cbfd3ae55e,
tests d2fd10267522bd1d3431336c77e05453f501f00ea1c78a7c4df7d53637e14ab3.
