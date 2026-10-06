# 015 — TRAIN-only fine-tuning последнего блока ResNet18

Статус: open — library приёмка подтверждена, публикация PR ожидается

## Проблема и целевое изменение

Frozen ImageNet features + nested fit-only logistic regularization дали TRAIN-only
mean foldAUROC0,688. Следующая одна гипотеза — адаптация layer4 convolution weights
к authorROI при сохранении wholeROI transform/mean16. Это отдельная library;
CPU extractor и scratch trainer defaults не меняются. Private orchestration
владеет split/selection/cloud staging; new API сам не выбирает пациентов.
Связь FR-ML-OSSEOUS-FEATURES / FR-ML-OSSEOUS-DEVELOPMENT.

## Приёмка

- Только локальный digest-pinned ImageNet ResNet18, offline loading. Stem/layer1–3
  frozen. Все BN affine/running stats, включая layer4, frozen/eval и после train().
- WholeROI96→224bilinear/antialias/RGB/ImageNetnormalization,16-slice mean;
  reusable frozen prefix FP32cache [N,16,256,14,14] без labels/fittedstats.
- Zero binary Linear512→1head. Первые2эпохи onlyhead, затем layer4 conv+head.
  AdamW headlr1e-3/conv1e-5,weightdecay1e-4,clipnorm1,batch4,seed42;
  pos_weight только по переданным fit-targets, обе категории обязательны.
- Явные FP32CPU/synthetic и BF16CUDAtraining (capability check без silentfallback);
  logits/BCE и predictionsFP32. Строгие finite/shape/target/epoch проверки,
  stable safeerrors; caller deadline/observer, finitegradients/loss.
- Synthetic boundedfit: frozenBN/prefix, warmup/head/layer4 updates,
  permutationinvariance/direct-cache equivalence и state_dict replay.
  Независимое review/tests/speccheck; платный job этим change не подтверждается.

## План

1. Прочитать existingAPI, red tests, bounded new layer4 module.
2. Green focused tests и independent review, permanentcontract/nav.
3. Отдельный PR; preregistered private worker/staging/research evidence отдельно.

## Результаты

2026-10-06: реализация и library приёмка завершены; пункт3 — публикация PR — ожидается. Module288/test261строк.
31focusedsyntheticchecks peerpassed2,58s;author62adjacentchecks2,74s;
Rufflint/formatpassed. Independent review accepted sourceSHA
 ec63dafb3bc86b8d68208ce23271403ace1d6ab85f967936cc325067cef8064f,
 testsSHA78dc4a0758ea20f31699947967b16ebc274d89c07a1b7d6251bc44489f8cbb72.

Root immutableprivatev2bundle CPU smoke:2realTRAINbags,3эпохи,exactreload,
2,40s,testinferencefalse. SyntheticTERMwatchdog:childgroupkilled/reaped,
parentexit130/cancelledreceipt. Privateworker/staging/bootstrap reviewed отдельно;
throughput/runtime остановки сохраняют classification, lifecycleTERM/INT cleanup.
FR/TC-ML-OSSEOUS-LAYER4, traceability и navigation обновлены.

Техническая приёмка относится к library. Preregistered5foldGPUэксперимент и его
результаты ведутся отдельно; source review/CPU smoke не доказывают улучшение
метрик, CUDA выполнение или clinical readiness. Patientdata/weights приватны.

`python3 scripts/check_specs.py` и `--self-test` →passed.
