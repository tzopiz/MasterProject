# 006 — Воспроизводимое обучение костных изменений

Статус: in-progress. Основание: [005](../005-od3d-preparation/README.md).

## Контракт и план

1. training/tmj_osseous_research.py принимает готовый indextask
   tmj-osseous-author-roi-v1; mode binary либо multilabel(порядок1..6).
   DataLoader возвращает[B,K,1,H,W], CNN2D sharedper-slice+meanpool,
   binaryone logit либо6independentlogits. Не использоватьpositionhead.
2. Frozen split по source-qualified patient_id, proportions70/15/15 seed42,
   сохранитьprivatemembership+digest; обе сторонывсегдавместе.
   Выбордетерминированный,каждыйpartitionсодержитобаbinaryclasses;
   testникакнеучаствуетвэпохах,весахloss,порогах. Непригодныйsplitотклонять.
3. BCEWithLogitsLoss; classweightsизtrain, AdamW lr0.001,
   max40epochs, earlystopping8, batch4, seed42, noaugmentationbaseline.
   Выборэпохи valAUROC(binary)/valmacroAUPRC(multilabel),
   порогYoudenJvalidation(binary), per-labelvalidationthresholds(multilabel).
4. Baselineconstanttrainprevalence. HoldouttestAUROC/AUPRC,
   confusion/sensitivity/specificity и patientbootstrapCI200draws(seed42).
   Aggregatepublicreport, predictions/split/privatecheckpoint metadata;
   artifactcompletiontarget/codebook/input/splitdigests, checkpointreload.
5. tools/run_osseous_research.pyconfigCLIpreflight/train;
   existingpositionrunnerнеизменяется. Cloudtoolстроитотдельныйpayload/job
   с темже intake/split, python3.12/pinnedrequirements/watchdog;
   запускпослеконкретногоподтвержденияресурсов. Libraryvalidationreused,
   sagittalcanonicalserializationнеиспользуется.
6. TestsmeaningfultinyCPUtrain, patientleakage, testlabelperturbationне меняет
   selectedepoch/threshold, sourceartifactmutations, binary/multilabel
   checkpoint reload, cloudpreparezeroexternalIO.

## Приёмка

РеальныеданныеpreflightPASS,локальныйsmokePASS,preparedbundlePASS,
конкретныйDataSpherejob/project/resource/runtime/priceplanготов.
Улучшениеметрикнеобещаетсядообучения; готовностьзапускадолжнабытьпроверена.

## Фактические проверки

22 focused tests пройдены: оба режима обучения, reload checkpoint, frozen
split и отсутствие влияния test labels на selection. Дополнительно те же
22 теста пройдены в изолированной среде Python 3.12.14 / Torch 2.6.0 CPU;
`pip check` не обнаружил конфликтов. Реальный GPU не запускался.
Полный реальный preflight остаётся открытой частью приёмки.

Полный ML-набор после интеграции: `python -m pytest MLService/tests -q` —
534 passed за 35.44 s в cloud-pins-venv; проверка локальная, без paid jobs.

На двух реальных авторских ROI из development cohort прошли forward, BCE loss,
backward и optimizer step обоих режимов на Torch 2.6.0 CPU. Полезность модели
по этим двум примерам не оценивалась. Первые три технически просмотренных
пациента сохранены в приватном development-only списке для исключения из test.


### Уточнения после независимой проверки trainer

Frozen membership/digest записывается в `split.private.json` сразу после
создания приватного output directory, до модели, optimizer и первого шага.
Fault injection в первый optimizer step подтверждает сохранение того же digest;
checkpoint и completion при прерывании отсутствуют.

Для каждого выхода сохраняются `train_supported_label_mask`,
`validation_supported_label_mask`, их пересечение
`evaluation_supported_label_mask`, `threshold_sources` и `label_limitations`.
При отсутствии train или validation class support порог 0.5 явно отмечен
fallback. Такие выходы сохраняют probabilities/confusion, но не входят в
aggregate test AUROC/AUPRC/bootstrap; их per-label AUROC/AUPRC равны null.
Mask и выбор порогов не используют test labels. Правило binary partitions,
модель, loss weights и hyperparameters не менялись.

Два новых regression tests сначала воспроизвели отсутствие раннего split
и validation support metadata. После исправления в существующем Python 3.12
`cloud-pins-venv` из `MLService` с repository/MLService в `PYTHONPATH` выполнено
`python -m pytest tests/test_tmj_osseous_research.py -q -p no:cacheprovider`:
**24 passed in 3.71s**. Смешанный multilabel fixture отдельно проверяет
train-supported/validation-unsupported и train-unsupported/validation-supported
выходы, fallback sources, исключение из aggregates и checkpoint reload metadata.
Общий ML suite и реальное обучение в этой ограниченной проверке не запускались.


Preflight теперь отклоняет multilabel конфигурацию без хотя бы одной метки,
имеющей оба класса одновременно в train и validation, кодом
`no_validation_supported_labels`. Общая pure target mapping используется и
preflight, и DataLoader; test labels не участвуют в этом gate. Существующий
regression test сначала показал ошибочный ready, затем подтвердил отказ без
создания output. После изменения focused suite в том же Python 3.12 окружении:
`python -m pytest tests/test_tmj_osseous_research.py -q -p no:cacheprovider` —
**24 passed in 3.36s**. Добавлено 5 net production lines и 3 assertion lines;
model/hyperparameters и прочие компоненты не менялись.

Независимое повторное ревью подтвердило устранение замечаний по раннему
frozen split и class support; регрессий в ограниченном review не найдено.
После всех исправлений: полный ML suite — **585 passed in 41.01s**;
trainer/cloud на отдельном Python 3.12.14 / Torch 2.6.0 CPU —
**44 passed in 6.69s**. Эти локальные проверки не заменяют полный real-data
preflight и не подтверждают улучшение метрик.
