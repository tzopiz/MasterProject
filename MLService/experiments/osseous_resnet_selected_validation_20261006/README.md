# TRAIN-selected regularization: validation — 2026-10-06

Статус: completed

## Вопрос и критерий решения

После успешной [nested TRAIN-only CV](../osseous_resnet_nested_regularization_20261006/README.md)
выбрать C по тому же grid на full TRAIN и проверить замороженный classifier.
Grid=[1e-5,1e-4,1e-3,1e-2,1e-1,1]. Sorted TRAIN patient keys,
StratifiedKFold3 по joint-status,shuffle=True,seed42; mean3foldAUROC максимум,
точная ничья — меньший C. Затем refit204TRAIN sides с fit-only scaler/balanced weights.
Originalvalidation не выбирает C или threshold. Threshold фиксирован0,5.

Go сохраняет исследовательский кандидат при validation AUROC≥0,65,
AP≥constant train-prevalence AP+0,05 и supported source151AUROC≥0,5.
Это тот же исследовательский admission criterion предыдущей frozen C1 проверки;
он не означает clinical acceptance. Пациентский paired AUROC difference относительно
C1 — secondary, без выбора hyperparameter по нему. Если no-go, остаётся C1.

## Данные и изоляция

Те жеTRAIN102patients/204sides137positive/67normal и validation22patients/43sides
32positive/11normal. Index SHA256
`f2aa5b44e3931dfe88a6c16f08c12cd916ba58256766c5b39da02709e728f235`;
original membership digest
`c65cb2ef35e42b042b4607d33dc0c387b57021ea8309c6cee42eab1d38a418bc`.
Hash-checked frozenResNet512feature caches; order проверяется по прежним
patient/side/target receipts. Pixels не загружаются повторно. Sourceprofiles по
pinned authorCSV. Weight/transform совпадение TRAIN/validation cache обязательно.
Test membership только для disjointness; test payload полностью исключён.

## Воспроизведение и ресурсы

Python3.12.14,NumPy2.1.3,sklearn1.6.1,CPU;≤60s для полного fit/metrics/bootstrap.
Новые downloads, платные jobs и обучение backbone отсутствуют.
Каждый fit — fresh StandardScaler+LogisticRegression(Cselected,balanced,liblinear,
max_iter2000,seed42), ConvergenceWarning прекращаетrun.18selection fits+1refit.
До первого fit сохранить memberships/grid/selection manifest; до чтения validation
feature values сохранить selectedC,checkpoint,threshold,source artifact hashes.
Checkpoint reload проверяется точно на TRAIN иvalidation probabilities.
Private entrypoint research-20261005/resnet-selected-validation-20261006/run.private.py.

## Оценка и ограничения

AUROC/AP,confusion/sensitivity/specificity/balancedaccuracy при0,5,
sourceprofile support и metrics (oneclass=null); constantbaselineTRAIN prevalence.
200patient bootstrap draws,оба сустававместе; intervals AUROC/AP/balancedaccuracy и
paired AUROC difference vsC1,valid count явный. CI не учитывает selection/retraining.

Это повторно используемая exploratory validation: результаты C1 на ней уже известны.
NestedCV и fullTRAIN selection снижают внутреннюю утечку C, но не восстанавливают
независимость validation. AuthorROI и sourceconfounding ограничения сохраняются.
Связь FR-ML-OSSEOUS-DEVELOPMENT / FR-ML-OSSEOUS-EVALUATION.

## Результаты

18selectionfits и1fullTRAINrefit выполнены за0,30s. TRAIN-only selection
выбрала C=0,01; порог0,5. Convergence/reload/hashes/resources проверены.

| Показатель | TRAIN-selected C0,01 | Прежний C1 |
|---|---:|---:|
| Validation AUROC | 0,727273 | 0,732955 |
| Validation AP | 0,895657 | 0,894252 |
| Balanced accuracy при0,5 | 0,659091 | 0,646307 |
| Sensitivity / specificity | 0,5 / 0,818182 | 0,65625 / 0,636364 |
| Profile151 AUROC | 0,571429 | 0,571429 |

Confusion TN9,FP2,FN16,TP16. Profile11125positive/0normal,AUROC/AP=null;
profile1517positive/11normal,AP0,483248,balancedaccuracy0,623377.
200/200patient-bootstrap:95%CI AUROC[0,545227;0,878717],
AP[0,771197;0,978808],balancedaccuracy[0,526460;0,788054].
PairedAUROC difference−0,005682,CI[−0,182273;+0,219007]. Validationranking
заметно не улучшился; небольшое улучшение balancedaccuracy не подтверждает
устойчивое превосходство. Sensitivity снизилась — threshold tradeoff явный.

Admissioncriterion выполнен; сохранён TRAIN-selectedresearchcandidateC0,01.
Program SHA256 `4aaf6fbf3b845578951b4905d694e9e251163dd2dd327510245de0ebb5befd8c`.
Manifest SHA256 `066501c0805636761f3eed1928e273b8f648058ee0560f27642c05f1f0cec15e`.
Private checkpoint SHA256 `b92b9c4134c7e85bfd41d5f748e00100fbd999c5bd494d878e5307566ce41c12`.
Веса/predictions/features/selectionmembership остаются приватными. Test не читался.

## Вывод

При go сохранить TRAIN-selected head и замороженный recipe. Следующая задача —
воспроизводимый repo entrypoint и отдельная multilabel проверка support/метрик.
Test не используется для дальнейших изменений модели.
