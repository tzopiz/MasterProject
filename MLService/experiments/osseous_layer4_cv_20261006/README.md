# Osseous layer4 TRAIN-only CV —2026-10-06

Статус: completed, negative adoption decision. Протокол ниже был зафиксирован до обучения; фактические результаты приведены в конце.

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
review обязательны до cloudsubmit. Платный запуск завершён; критерий improvement не выполнен.


## Результат A100 — 2026-10-06

DataSphere Job `bt1ovdhqt335unrmvb7q`: provider SUCCESS, worker complete/exit0.
Фактическая среда: NVIDIA A100-SXM4-80GB, PyTorch2.6.0+cu118, CUDA11.8.
Завершены все пять outer folds; selected epochs `[16,16,12,16,8]`,
5236 optimizer steps. Исходные validation/test не открывались.

| Метрика | Layer4 | Frozen nested logistic comparator |
|---|---:|---:|
| Mean fold AUROC | 0,697102 | 0,687931 |
| Mean fold AUPRC | 0,846865 | 0,833053 |
| Pooled OOF AUROC | 0,672513 | 0,669899 |
| Profile111 pooled AUROC, 82 стороны | 0,714962 | 0,748106 |
| Profile151 pooled AUROC, 122 стороны | 0,645954 | 0,611433 |

Fold AUROC: `[0,720988;0,775510;0,693452;0,592262;0,703297]`.
Парная meanfold разница +0,009170; выигрышей4/5. Paired patient bootstrap,
200 valid draws: 95% interval разницы `[-0,025190;+0,049245]`.
Profile111 support38positive/13negative contributingpatients;
profile15146positive/36negative. Категории могут пересекаться для пациента
с нормальной и патологической сторонами.

**Решение: retain_frozen_logistic.** Прирост меньше заранее заданных0,025,
а regression profile111−0,033144 превышает допустимые0,02. Пороговые метрики
не используются для отмены этого решения: при0,5 pooled TN29/FP38/FN38/TP99,
BCE0,916530; BCE по обоим профилям хуже comparator. Новая модель не выбрана
для originalvalidation; эти development folds не дают новой blind holdout оценки.

Worker80,05с; весь provider lifecycle394,693с (около6,6мин). Измеренные
120actualupdates:48,06updates/s, peakCUDA336931840bytes. Оценка вычислений
по опубликованной ставке542,88RUB/hour:59,52RUB за lifecycle; фактический
счёт не проверен, storage/egress не включены. Начальный forecast и reservation
выше — ограничения до запуска, а не фактическая стоимость.

## Проверка артефактов

Приватный completion содержит hashes всех артефактов; сверены все hashes,
в том числе11обязательных файлов. Для всех folds повторно вычислены AUROC,
сверены identities/targets с TRAIN manifest и inner epoch selection до outer
inference. Все5producer checkpoint reload checks успешны. Дополнительно CPU
replay fold1 на PyTorch2.11.0: max probability difference1,2517e-6 (<1e-5),
AUROC совпал. Никакие patient IDs, predictions, cache или веса не опубликованы.

- TRAIN manifest SHA256: `bba8660db1e1186a89a7411a8979312d664ddce61d636288b00e9376fb20eb61`.
- Aggregate report SHA256: `e43c3c09bc2df63ce359b693483172b8c14596ff2cbe1abd353a83231d6eafc4`.
- Completion SHA256: `fce5f56359d472cdcdc42027e2ed5fc793b521ec13095761519c5d7c2e9c8aab`.

Следующая гипотеза оформляется отдельным протоколом. Отрицательный результат
этого завершённого эксперимента не означает готовность произвольного DICOM
к классификации или завершение общей исследовательской задачи.
