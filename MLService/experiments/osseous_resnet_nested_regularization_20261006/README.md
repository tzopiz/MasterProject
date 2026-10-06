# Nested TRAIN-only regularization — 2026-10-06

Статус: completed

## Вопрос и критерий решения

Уменьшит ли nested selection силы регуляризации переобучение512-feature head?
Frozen C1 достигает TRAIN AUROC1 во всех фолдах, но mean outer AUROC0,609.
Сравнить learning algorithm с выбором C исключительно во внутренних patient-folds
против фиксированного C1 на прежних пяти outer folds.

Primary mean5outerfoldAUROC. Go: все nested fits converged/completed;
mean paired improvement≥0,02; строгое улучшение в≥4/5outerfolds;
pooled profile111/151 AUROC≥0,5. Иначе оставить прежний fixed C1 research candidate.
Порог0,02 относится к этому исследованию, не clinical acceptance или significance.
Не выбирать лучший outer C и не менять grid после просмотра результатов.

## Данные и разбиение

Только прежние frozenTRAIN102patients/204sides. Image features из hash-checked
ResNet18 mean16 cache512dim; source/index/order/weights рецепт неизменен.
Outer fold digest `4319716f75d907f1cba37ffdd9cfdc41c9fa523b1b6196bdb82f0da00cccc033`.
Index SHA256 `f2aa5b44e3931dfe88a6c16f08c12cd916ba58256766c5b39da02709e728f235`.
TRAIN-only feature archive и prior report/results/predictions/manifest hashes
закрепляются в новом manifest; comparison scores пересчитываются и сверяются по
явному fold number. Patient key order сверяется до использования feature values.
Original validation/test изображения, features и predictions не открываются.

Для каждого outer fit: sorted patient keys, StratifiedKFold3 по patient joint
status(sum binary targets двух сторон), shuffle=True,seed=100+outerfoldnumber.
Оба сустава вместе; outer evaluation patient не входит ни в один inner partition.
Все outer/inner memberships сохраняются ДО первого fit. Coverage каждой стороны
outer-evaluation ровно1раз; обе бинарные категории в каждом inner fit/eval обязательны.

## Модель, выбор и ресурсы

Grid C=[0,00001;0,0001;0,001;0,01;0,1;1]. Каждый fit — новый StandardScaler и
LogisticRegression(balanced,liblinear,max_iter2000,seed42) только на fit samples.
Выбор максимального среднего3innerfoldAUROC, tie-break меньший C. После выбора
refit полного outer fit и однократная оценка его outer evaluation.
Нет PCA, sourcefeatures, новых patients, pixels, pretrainedweights или cloudjobs.

Python3.12.14,NumPy2.1.3,sklearn1.6.1,CPU; общий wall≤180s,
ConvergenceWarning останавливает run. Private entrypoint
research-20261005/resnet-nested-regularization-20261006/run.private.py.
Сохраняются hashes программы/manifest/source artifacts, inner tables, checkpoints,
OOF predictions и private report. Exact checkpoint reload probabilities обязательны.

## Оценка и ограничения

Mean outerAUROC/AP, pooledOOF metrics, sourceprofile support/metrics, выбранные C
по каждому outer fold. Paired200patient-bootstrap draws внутри outerfolds,
обе стороны вместе, те же draws comparator; valid count/null явно.
CI не учитывает variability retraining. Это exploratory nested development CV:
выбор самого метода и grid мотивирован уже просмотренными TRAIN/validation
результатами, поэтому outer оценка не превращается в независимый финальный test.
Новый C не выбирается на originalvalidation. Связь FR-ML-OSSEOUS-DEVELOPMENT /
FR-ML-OSSEOUS-EVALUATION.

## Результаты

Все90inner и5outer fits завершены за0,89s; convergence/reload checks прошли.
Выбраны C по outerfolds:0,01;0,001;0,001;0,01;0,001.

| Показатель | Nested selection | Fixed C1 |
|---|---:|---:|
| Mean outer AUROC | 0,687931 | 0,608655 |
| Mean outer AP | 0,833053 | 0,788399 |
| Pooled OOF AUROC | 0,669899 | 0,612921 |
| Profile111 AUROC | 0,748106 | 0,639205 |
| Profile151 AUROC | 0,611433 | 0,559514 |

Mean paired AUROC difference+0,079276; strictly positive4/5folds.
200/200pairedpatient-bootstrap draws:95%CI[+0,027765;+0,138888]. Это descriptive
интервал фиксированных predictions, не retraining/model-selection uncertainty.
Criterion выполнен; выбор на full TRAIN разрешён. Outer fold3 ухудшился на0,050595.
Program SHA256 `d1c0fbaa30d7892f4a6240252466fdb5f5a72aa0cdf04a3a19ae0ebc938f3605`.
Manifest SHA256 `10762c481a01c6042a3abf10f29a4a84cb896ce495ef418946d32035c9f42d36`.
Hashes/program/convergence5folds/reload/resource проверены. Изображения и original
validation/test feature/prediction payload не читались. Все private artifacts ignored.

## Вывод

При go — отдельный протокол выбора C на full TRAIN через3patient folds и refit
полного TRAIN, затем замороженная validation проверка. При no-go fixed C1 остаётся.
Любая finaltest оценка требует отдельного freeze/protocol; test повторно не открывается.
