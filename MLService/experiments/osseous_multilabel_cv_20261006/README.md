# Frozen mean16 — multilabel capability CV, 2026-10-06

Статус: planned. Протокол фиксируется до нового fit; предварительно проверены
только TRAIN label counts и partition support. Prediction quality пока не вычислялась.

## Вопрос и границы

Оценить шесть независимых osseous выходов на существующих frozen mean16 features.
Это отдельный multilabel путь: бинарный nested mean16 кандидат остаётся прежним;
max шести scores не объявляется вероятностью OR. Сочетания патологий сохраняются.
Связь FR-ML-OSSEOUS-LABELS / FR-ML-OSSEOUS-FEATURES /
FR-ML-OSSEOUS-EVALUATION / FR-ML-OSSEOUS-DEVELOPMENT;
[изменение017](../../../docs/changes/017-osseous-multilabel/README.md).

## Данные и источник

TMJ-OD3D V2 authorROI, bounded cohort146patients/289sides, source complete=false.
Только исходный TRAIN102patients/204sides. Code order1–6: cortical erosion,
subchondral sclerosis, subchondral cystic changes, condylar flattening,
osteophyte, Other. Explicit `[0]`→шесть нулей, pathology combinations→multi-hot.
Invalid/missing/conflicting labels не превращаются в normal; OR targets должны
точно совпасть с прежними binary labels. Counts положительных сторон:
`[77,14,20,29,36,9]`, разных positive-contributing пациентов:`[55,13,16,23,29,7]`.

Index SHA256 `f2aa5b44e3931dfe88a6c16f08c12cd916ba58256766c5b39da02709e728f235`;
original split digest `c65cb2ef35e42b042b4607d33dc0c387b57021ea8309c6cee42eab1d38a418bc`;
mean16 feature cache SHA256 `316bc53a33222896035c6c897845aa5a7273d66ea549454714c9abe9064d635a`.
Прежние5outer/3inner patient memberships из nested manifest SHA256
`10762c481a01c6042a3abf10f29a4a84cb896ce495ef418946d32035c9f42d36`
используются без пересоздания. Обе стороны вместе, no leakage; outer coverage1.
Никакие originalvalidation/test payloads или изображения не читаются.

## Модель и selection

Offline существующий feature/head API, mean16 recipe без нового extraction.
Один общий C для всех6 independent balanced liblinear heads; grid
`[1e-5,1e-4,1e-3,0.01,0.1,1]`, max_iter2000,seed42. Каждый fit — fresh fit-only
StandardScaler + heads. One-class fit label→fit prevalence constant и supportfalse.

Для каждого outer fit ДО любого fit определить один selection mask: label имеет
обе категории во всех3 inner fit/evaluation partitions. Этот mask одинаков для
всех C, не зависит от outer evaluation labels. Пустой mask→inconclusive, без fit.
Выбор C: максимальный mean3inner macro AUPRC по этому mask, tie→меньший C.
После выбора fresh outer refit и единственный outer inference всех6 outputs.
Сохранить отдельные inner/outer-fit support masks и причины отсутствия поддержки.
Threshold0,5 — только confusion diagnostics, без tuning/calibration;
balanced scores не объявляются calibrated probabilities.

## Оценка и ограниченная support

Evaluation mask = selection mask ∩ outer fit support ∩ outer evaluation class
support. Последний компонент используется только для undefined metrics после
фиксирования C/predictions. Unsupported outputs не входят в aggregate metrics,
их AUROC/AP null; probabilities и confusion diagnostics остаются. Для каждого
label/partition указать positive/negative стороны и distinct contributing patients.

Primary: mean5outerfold macro AUPRC, paired difference к constant train-prevalence
baseline на ТОЧНО том же eligible mask каждого fold. Публиковать каждый mask,
его denominator и scores: состав macro может различаться, это не all-six quality.
Secondary: macro AUROC, foldspread, per-label AP/AUROC, confusion, sourceprofile
support. Pooled per-label считать на eligible OOF rows; требуется≥5positive и
≥5negative contributingpatients. Иначе AP/AUROC/CI null и вывод inconclusive.
Указать excluded rows и pool population, не выдавать subset за всю когорту.

Paired200patient-bootstrap draws внутри5outerfolds (обе стороны вместе),seed42.
Primary draw valid только при обеих категориях на каждом label фиксированного
fold mask; mask не сокращается в bootstrap. Pooled per-label draws проверяются
отдельно на исходном фиксированном pool mask. 95% percentile interval только
при≥50validdraws, иначе null с причиной; denominator/validcount сохраняются.
CI фиксированных OOF scores не учитывают retraining variability.

Exploratory signal flag требует complete5folds, mean paired macroAP gain≥0,05,
wins≥4/5 и pooledAUROC≥0,5 для всех globally supported outputs. Это лишь signal
на поддержанном subset, не all-six acceptance и не разрешение открыть старую
validation/test. Для следующего запуска нужен новый протокол. Эта CV является
development evidence на reused folds, а не новым blind holdout.

## Ресурсы и артефакты

Локальный CPU,Python3.12.14/NumPy2.1.3/sklearn1.6.1,2threads,seed42.
Cap180s для fits/bootstrap, не более90inner+5outer bundle fits (≤570estimators).
Convergence/nonfinite/hash/membership/target failure→technicalfailure, без success.
Без download/GPU/cloudjobs. Source/cache/index/code/protocol/library hashes и
membership/support masks сохраняются до первого fit в private manifest.
Checkpoints точно воспроизводят OOF scores, code order и metadata.
Private root:`research-20261005/resnet-multilabel-cv-20261006/`, entrypoint
`run.private.py`; files600/directories700. Public source API не меняется.

## Результаты и приёмка

До запуска: не запускалось, метрик нет. Обязательны независимое worker/protocol
review, verified hashes/checkpoint replay и aggregate result review. Завершение
исследования допустимо при отрицательном signal или inconclusive rare labels;
техническая capability не доказывает диагностическую пригодность всех6 outputs.
Weights/features/IDs/individual predictions не публикуются.
