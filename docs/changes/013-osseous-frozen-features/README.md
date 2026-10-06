# 013 — Reusable frozen osseous features

Статус: open

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

2026-10-06: пункт1 поручен отдельному gpt-6.1-sol/high агенту; only two new
module/test paths, scope≈400handwrittenlines; no CLI/cloud/privatecohort work.
Приёмка пока не подтверждена. Приватные research результаты доступны только
агрегатами в [реестре экспериментов](../../../MLService/experiments/README.md).
