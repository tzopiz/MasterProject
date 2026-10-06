# Paired osseous TRAIN-only grouped CV — 2026-10-06

Статус: completed

## Вопрос и критерий решения

Проверить перенос spatial-head capacity на новых TRAIN пациентах. Сравнить
исходную global CNN и spatial-head(4x4,256→32→1) с нуля на одинаковых5folds.
Primary: равновесное среднее5foldAUROC. Secondary: mean fold average precision,
paired differences, pooledOOFAUROC/AP и profile111/151 support/metrics.

Выбрать spatial для следующего validation experiment только если одновременно:
все10runs завершили40epochs; mean paired AUROC difference>=.05; difference>0
минимум4/5folds; pooledOOFAUROC>=.5 отдельно внутри111 и151. Ничья не выигрыш.
Это development go/no-go, не clinical target или statistical significance gate.
При отрицательном результате не выбирать candidate по отдельным удачным фолдам.

## Данные и разбиение

FrozenTRAIN102patients/204sides,67normal/137pathology из выбранной TMJ-OD3D V2
authorROI когорты. Originalvalidation/test crops/predictions не открываются.
Индекс SHA256 f2aa5b44e3931dfe88a6c16f08c12cd916ba58256766c5b39da02709e728f235;
global split SHA256 c65cb2ef35e42b042b4607d33dc0c387b57021ea8309c6cee42eab1d38a418bc.

Patient-status counts: normal18,discordant31,pathology53. Status×sourceprofile
шесть strata infeasible(normal111=3<5): заранее задан fallback на3statuses.
Sorted TRAIN patient keys; StratifiedKFold(5,shuffle=True,random_state42) по
patient joint-status. Patient sides остаются вместе, каждый оценивается1раз.
Assignments фиксируются приватным manifest до обучения; folds не перерисовываются.

| Fold | Fit/Eval patients | Eval normal/positive sides | Eval profile111/151 patients |
|---|---:|---:|---:|
| 1 | 81/21 | 15/27 | 9/12 |
| 2 | 81/21 | 14/28 | 8/13 |
| 3 | 82/20 | 12/28 | 8/12 |
| 4 | 82/20 | 12/28 | 6/14 |
| 5 | 82/20 | 14/26 | 10/10 |

AuthorCSV selected_slice_count используется только для stratification audit/
subgroup reporting, не передаётся модели. Fold4/profile111 не имеет normal:
его ranking metrics=null, причины/support фиксируются; pooled111 поддержан16/66,
pooled151 поддержан51/71. В baseline constant fit-prevalence AUC=.5.

## Ресурсы и воспроизведение

Python3.12.14,Torch2.11.0,NumPy2.1.3,sklearn1.6.1,CPU2threads,seed42,
deterministic. Author16x96x96 float16 bags→float32 в памяти; existing normalization.
Каждый model/fold40epochs, batch4,AdamW lr=.001,weight_decay=.01; fold-fit-only
pos_weight. Paired identical seeded DataLoader order, independent fresh models.
Без early stopping/augmentation; eval только epoch40. Fit images сверяются с crop
SHA256 до загрузки. Общий wall1200s; incomplete протокол не сравнивает разные бюджеты.
Ignored grouped-cv-20261006 хранит программу/fingerprint,manifest,progress,
private predictions/metadata,aggregate report. Новых платных jobs и загрузки данных нет.

## Оценка и ограничения

Mean/SD/range5foldmetrics; paired mean difference CI от200resamples пациентов
внутри каждого evaluation fold (оба сустава, identical draws двух моделей).
CI descriptive: не включает variability retraining/overlapping fit sets.
Все null/invalid draws явны; average precision не trapezoidal PR area.
PooledOOFscore scales могут различаться между folds, это secondary.
Capacity probe и выбор архитектуры используют TRAIN, поэтому exploratory
development estimate. Originalvalidation/test вне сравнения; frozen originalsplit
не изменяется. Source confounding не устранён одной sanity-проверкой>=.5.
FR-ML-OSSEOUS-DEVELOPMENT / FR-ML-OSSEOUS-EVALUATION.

## Результаты

Все10runs завершили40epochs,CPUwall524.01s<1200s.
Frozen fold digest4319716f75d907f1cba37ffdd9cfdc41c9fa523b1b6196bdb82f0da00cccc033.
Program SHA256406e4e5a871571ace938280edb72feb82b7ef794d3bd7d3dff244efecb1c1174.

| Модель | Mean fold AUROC ±SD | Mean fold AP | Pooled111/151 AUROC |
|---|---:|---:|---:|
| global baseline | .5054 ±.1083 | .7225 | .4337 / .4769 |
| spatial head | .5197 ±.0750 | .7590 | .4943 / .4791 |

Paired differences по folds: .09383,-.09949,.10714,.00000,-.03022.
Mean difference .01425, wins2/5; paired patient-bootstrap200/200:
descriptive95%CI[-.07139,.11729]. Profile111/fold4 unsupported/null, как запланировано.

Программа до использования результатов прошла независимый static review:
TRAIN ownership, ordering/targets, fold-only pos_weight, fixed40epochs и go/no-go
корректны. Deadline проверяется в train loop, но не на inference/bootstrap:
для этого запуска final wall ниже cap; будущий runner должен расширить watchdog.
CSV runtime hash не был в manifest; отдельный аудит подтверждает byte SHA256
равен index.metadata_sha256 и source mtime раньше manifest creation.
Snapshot trainer проверен против main ceacc86 и сохранён приватно.
Private predictions/records/weights не публикуются; originalvalidation/test не использованы.

## Вывод

Go/no-go FAILED: spatial не выбирается для originalvalidation.
Успешный subset fit не перенёсся на новых пациентов; scratch CNN вероятно
не извлекает достаточно устойчивого сигнала на данном budget/cohort. Это
ограниченный эмпирический результат, не доказанная причина плохого finaltest.
Следующий [frozen ResNet18 experiment](../osseous_resnet_features_cv_20261006/README.md)
выполнен отдельно на этих же folds. В рамках этого scratch-CNN сравнения
original validation/test оставались закрытыми; позднее validation проверена
другим протоколом, а test для выбора модели не используется.
