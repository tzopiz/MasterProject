# Frozen ResNet18 max16 — TRAIN-only CV, 2026-10-06

Статус: planned. Протокол зафиксирован до извлечения новых признаков и fit.

## Гипотеза и границы

Coordinate-wise max по 16 frozen slice embeddings может сохранить локальные
костные признаки, которые усреднение ослабляет. Меняется только pooling;
размерность 512 и head остаются прежними. Max embeddings не является максимумом
вероятности патологии; возможно усиление артефактов и source signatures.
Связь FR-ML-OSSEOUS-FEATURES / FR-ML-OSSEOUS-DEVELOPMENT /
FR-ML-OSSEOUS-EVALUATION; [изменение 016](../../../docs/changes/016-osseous-max-pooling/README.md).

## Данные и изоляция

Та же bounded TMJ-OD3D V2 authorROI когорта:146 пациентов/289 сторон,
source complete=false. Только исходный TRAIN:102 пациента/204 стороны,
137 патологических и67 нормальных. Метки — авторские binary any-code1–6;
пропуски не превращаются в норму. Старые validation/test pixels/features/predictions
не читаются. Результат относится к authorROI, не к локализации в произвольном DICOM.

Index SHA256 `f2aa5b44e3931dfe88a6c16f08c12cd916ba58256766c5b39da02709e728f235`;
original split digest `c65cb2ef35e42b042b4607d33dc0c387b57021ea8309c6cee42eab1d38a418bc`.
Без пересоздания прежние5 outer patient folds и3 inner patient folds из nested
mean16 manifest SHA256 `10762c481a01c6042a3abf10f29a4a84cb896ce495ef418946d32035c9f42d36`;
outer fold digest `4319716f75d907f1cba37ffdd9cfdc41c9fa523b1b6196bdb82f0da00cccc033`.
Обе стороны пациента остаются вместе; outer evaluation ровно один раз для каждой
стороны. Все memberships/hash bindings сохраняются до первого fit.

## Рецепт и ресурсы

Offline full-digest ImageNetV1 ResNet18, frozen eval/FP32. Прежний wholeROI
bilinear antialias96→224, RGB repeat, ImageNet normalization,16 slices.
Сохранить per-slice [204,16,512] FP32 features; сравнить mean16 с прежним cache
SHA256 `316bc53a33222896035c6c897845aa5a7273d66ea549454714c9abe9064d635a`:
maxabs≤1e-6, иначе остановить run до fit. Единственный кандидат:
`z[j]=max(encoded[0:16,j])`. Max фиксирован заранее, mean/max не выбираются по CV.

Тот же C grid `[1e-5,1e-4,1e-3,0.01,0.1,1]`; fresh fit-only StandardScaler,
balanced liblinear logistic, max_iter2000,seed42. C выбирается по mean3innerAUROC,
tie→меньший C; fresh outer fit после выбора. Unsupported классы/convergence warning,
несовпадение split/crop hashes или nonfinite — технический отказ, без метрик успеха.

Локальный CPU, Python3.12.14/Torch2.11.0/torchvision0.26.0/NumPy2.1.3/sklearn1.6.1;
2 CPU threads, seed42, deterministic algorithms. Без нового download/GPU/cloudjob.
Extraction cap900s, fits/bootstrap cap180s. Код/manifest/cache/checkpoints/
individual predictions приватны, permissions600/700. Private entrypoint:
`research-20261005/resnet-max-features-cv-20261006/run.private.py`.
Metadata явно содержит max16; существующий public mean16 API не изменяется.

## Критерий решения

Comparator — сохранённые nested mean16 OOF predictions. MeanfoldAUROC0,687931420;
profile111 pooled0,748106061, profile1510,611433306. Обязательны complete5folds,
mean paired gain≥0,025 и строго положительные выигрыши≥4/5. Каждый supported
profile (≥5 positive-contributing и≥5 negative-contributing пациентов):
pooledAUROC≥0,50 и regression к comparator≤0,02. Недостаточная support→inconclusive.

AUPRC, pooled metrics, fold spread, selected C и paired200patient-bootstrap
95% interval — диагностика. Bootstrap фиксированных predictions не учитывает
retraining variability. Original validation/test не открываются в этом эксперименте,
даже при go: следующий шаг требует отдельного протокола. При failure сохранить
отрицательный результат и прежнего кандидата; max не подбирать после outer scores.
Это development evidence на уже использованных outer folds, не blind holdout.

## Приёмка и результаты

До запуска: не запускалось; метрик нет. Independent private-worker review,
hash checks, exact checkpoint reload и aggregate result review обязательны.
После исполнения записать время, фактическую команду, hashes и решение здесь.
