# Frozen ResNet18: original validation — 2026-10-06

Статус: completed

## Вопрос и критерий решения

Проверить фиксированный binary-кандидат из [TRAIN-only CV](../osseous_resnet_features_cv_20261006/README.md)
на оригинальной validation-выборке. Кандидат допускается к сохранению как
исследовательская модель, если validation AUROC ≥0,65, average precision выше
constant train-prevalence baseline минимум на 0,05 и AUROC внутри профиля151 ≥0,5.
Эти пороги — критерий данного эксперимента, не клиническая приёмка проекта.
Не выбирать C, backbone, preprocessing или порог по результатам этой проверки.

## Данные и разбиение

TMJ-OD3D V2, прежняя bounded archive-prefix когорта146patients/289sides.
Полный TRAIN102patients/204sides:137positive/67normal.
Original validation22patients/43sides:32positive/11normal. Оба сустава пациента
остаются в одной группе. Index SHA256:
`f2aa5b44e3931dfe88a6c16f08c12cd916ba58256766c5b39da02709e728f235`.
Membership digest:
`c65cb2ef35e42b042b4607d33dc0c387b57021ea8309c6cee42eab1d38a418bc`.
Test membership используется только для проверки отсутствия пересечений;
его изображения и predictions не открываются. Source profiles по pinned CSV.

## Воспроизведение и ресурсы

Рецепт неизменен: frozen ResNet18 ImageNet1K_V1, wholeROI bilinear/antialias224,
RGB repeat, ImageNet normalization,512feature/slice,mean16slices.
Official weights SHA256:
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
TRAIN features повторно используются только при совпадении manifest/hash/order.
StandardScaler и LogisticRegression(C1,balanced,liblinear,max_iter2000,seed42)
обучаются на полном TRAIN. ConvergenceWarning — отказ. Модель и metadata
сохраняются ДО первого чтения validation pixels. Reload probabilities проверяются.

Python3.12.14,Torch2.11.0,torchvision0.26.0,NumPy2.1.3,sklearn1.6.1.
CPU2threads,deterministic,seed42;300seconds включая fit/inference/bootstrap.
Новых downloads/cloud jobs/paidtraining нет. Entry point — ignored private
`research-20261005/resnet-validation-20261006/run.private.py`; сохраняются
script/loader/weights/checkpoint/feature/index/split fingerprints. Исходники
load_crop из branch codex/osseous-capacity-probes; его отдельный hash в manifest
фиксирует функцию независимо от несвязанных незакоммиченных architecture edits.

## Оценка и ограничения

Primary AUROC/AP; confusion/sensitivity/specificity/balanced accuracy при
фиксированном пороге0,5. Secondary порог Youden J определяется по прежним
TRAIN OOF probabilities ДО validation и замораживается в manifest. Это не
гарантия calibration полного TRAIN head; secondary report отражает этот перенос.
Constant predictor — prevalence только TRAIN, primary threshold0,5.
200patient-bootstrap draws, обе стороны вместе; single-class draws пропускаются
с явным valid count. Оба sourceprofiles отражаются отдельно; для one-class111
AUROC/AP=null, отсутствие normal support не восполняется.

Это exploratory validation: эта выборка ранее использовалась для выбора эпохи
первого scratch baseline, а TRAIN CV уже влияла на выбор нового кандидата.
Bootstrap не устраняет selection bias/sourceconfounding. Вход — authorROI;
это не оценка всего DICOM-пути и не классификация положения ВНЧС.
Связь FR-ML-OSSEOUS-DEVELOPMENT / FR-ML-OSSEOUS-EVALUATION.

## Результаты

Выполнено за16,63s на CPU; classifier сошёлся, exact reload подтверждён на
TRAIN и validation. Hashes script/manifest/classifier/validationfeatures проверены.

| Показатель | Frozen ResNet + C1 | Constant baseline |
|---|---:|---:|
| AUROC | 0,732955 | 0,5 |
| Average precision | 0,894252 | 0,744186 |
| Balanced accuracy при0,5 | 0,646307 | 0,5 |
| Sensitivity / specificity при0,5 | 0,65625 / 0,636364 | 1 / 0 |

Confusion при0,5: TN7,FP4,FN11,TP21. Patient-bootstrap200/200:
95%CI AUROC[0,529847;0,891901], AP[0,753270;0,975113],
balanced accuracy[0,507113;0,788273]. Это uncertainty фиксированной модели,
без поправки на предыдущий подбор.

Profile111:25positive/0normal,AUROC/AP=null. Profile151:7positive/11normal,
AUROC0,571429,AP0,495465,balanced accuracy0,532468. Общая discrimination может
сильно зависеть от source mix; внутри supported sourcegroup сигнал слабее.

TRAIN-OOF Youden threshold0,975080: sensitivity0,3125,specificity0,909091,
balanced accuracy0,610795. Этот перенос порога не улучшил balance; для сохранённого
кандидата остаётся preregistered primary0,5, validation порог не подбирается.

Program SHA256: `c3249a4bd2dd61dcacfdfd905fc4804152e5a2d9e7a60cbf472edff205453a8b`.
Manifest SHA256: `33f0d2b383f960dc476122415fdc287b14c815908b39ae02b791ec15b23b5974`.
Private checkpoint SHA256: `1f21c9ba2979956ecd2f0def255226cf4d3f51d6e0db6fdeb7ccfadc84166cf6`.
Отчёт, feature cache, membership и веса остаются ignored/private.

## Вывод

Критерий выполнен: frozen C1 research candidate сохранён с primary threshold0,5.
Модель не объявляется клинически готовой и не оценивает локализацию/DICOM chain.
Далее отдельный nested TRAIN-only CV для более сильной регуляризации head;
validation не используется для выбора C. Test повторно не открывается.
