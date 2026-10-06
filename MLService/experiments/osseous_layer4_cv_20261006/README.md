# Osseous layer4 TRAIN-only CV —2026-10-06

Статус: planned; protocol frozen before new training.

## Гипотеза и данные

Дообучение только ResNet18 layer4 conv weights + zero binary head улучшит
пациентскую TRAIN-only CV относительно nested frozen-logistic comparator.
Та же bounded TMJ-OD3D V2 authorROI когорта,146patients289sides; source complete=false.
Используется только исходныйTRAIN102patients204sides. Старые validation/test
pixels и predictions не загружаются. AuthorROI не подтверждает DICOM localization.

IndexSHA256 f2aa5b44e3931dfe88a6c16f08c12cd916ba58256766c5b39da02709e728f235;
originalsplitmembership c65cb2ef35e42b042b4607d33dc0c387b57021ea8309c6cee42eab1d38a418bc;
прежние5outerpatientfolds digest4319716f75d907f1cba37ffdd9cfdc41c9fa523b1b6196bdb82f0da00cccc033.
Точные crop/weights/code/worker/comparator hashes закрепляются в приватном manifest
до запуска; публичных patient IDs/features/predictions/checkpoints нет.

## Модель и выбор эпох

Offline ResNet18 ImageNetV1 с fixed fullchecksum как FR-ML-OSSEOUS-FEATURES.
WholeROIresize224/RGB/ImageNetnormalization,mean16. Prefix stem/layer1–3 frozen;
все BN affine/stats frozen/eval. Layer4conv + Linear512→1 head; zerohead.
Без augmentation/dropout/slice sampling/weighted sampler/scheduler.
AdamW headlr1e-3,conv1e-5,weightdecay1e-4,betas0,9/0,999,clipnorm1,batch4,seed42.
Первые2эпохи onlyhead. BCEpos_weight currentfitnegative/positive. BF16 training,
FP32 prefixcache/inference/selection, FP32optimizer/parameters; CUDA BF16 gate.

В каждом outerfit один patient75/25inner split, seed42,stratificationanysidepathology;
классы обязательны. До16эпох, innerAUROC на4/8/12/16; tie→ранняя.
Затем свежие originalweights/zerohead, refit всего outerfit выбранное числоэпох
с теми же2warmupepochs, новые fit-only classweights. Один outer inference.
Outer folds уже участвовали в развитии гипотез: это development evidence,
не новый blind holdout. Frozen prefixcache использует только внешние weights,
не labels/statistics; допустим между folds, хранится приватноFP32.

## Метрики и правило решения

Comparator: сохранённые nested TRAIN-only frozen-logistic OOF scores
(meanfoldAUROC0,687931420). Fixedthreshold0,5 diagnostic.
Go требует complete5folds, meanpairedAUROC improvement≥0,025 и выигрышей≥4/5.
Supportedsourceprofiles:≥5разных positive-contributing и≥5negative-contributing
patients; pooledOOFAUROC≥0,50 и regression к comparator≤0,02 в каждом.
Недостаточная support→inconclusive. AUPRC,BCE,foldspread и pairedpatientbootstrap
CI200draws —диагностика, без требования significance/clinicalclaim.

## Ресурсы, контроль и артефакты

Один DataSphere g2.1/A10080GB Job. VerifiedOct6published542,88RUB/hourinclVAT;
communityallow/API подтверждены, freecapacityне гарантируется.
Worker cap45min, lifecyclecap90min;90mincomputeestimate814,32RUB,
setup/storage/egress отдельно. Это оценка, не hardmoneycap; budgetreservation
приватная. Первые100actualsteps фиксируют throughput/peakmemory; forecast
за worker cap→technicalinconclusive. Unknownsubmission не повторяется автоматически.

Private fixedmanifest/cache/checkpoints/innerhistory/OOFpredictions/progress,
aggregate report/completion with digests. CPU smoke и independentworker/staging
review обязательны до cloudsubmit. Платный запуск и improvement пока не выполнены.
