# Frozen ResNet18 features: TRAIN-only grouped CV — 2026-10-06

Статус: completed

## Вопрос и критерий решения

Могут ли pretrained image features дать более устойчивый сигнал, чем две scratch
CNN на том же TRAIN-only patient CV? Feature extractor ResNet18 ImageNet1K_V1
фиксирован и не обучается на TMJ. Mean pooling16slice512dfeatures;
StandardScaler+LogisticRegression(C=1,class_weight=balanced,solver=liblinear,
max_iter2000,random_state42). Scaler/head fit только на каждом fit fold.

Primary: mean5foldAUROC. Comparator — прежняя spatial(.519658 mean), лучший
из двух по этому summary; это выбор development comparator, не независимая оценка.
Go: все5fit завершены/converged; mean paired AUROC difference>=.05; строго>0
минимум4/5folds; pooledOOFAUROC внутри111 и151>=.5. Иначе candidate не выбирается.
Mean AP,allfoldmetrics,pairdifference bootstrap и source groups secondary.
Epoch/threshold/hyperparameter selection отсутствует. C не настраивается по folds.

## Данные и разбиение

Те же102frozenTRAINpatients/204sides и5foldmanifest grouped-cv-20261006,
fold digest4319716f75d907f1cba37ffdd9cfdc41c9fa523b1b6196bdb82f0da00cccc033.
Originalvalidation/test crops/predictions не открываются. Membership и pinned
index/split проверяются до pixels. Каждый patient evaluated1раз,оба сустава вместе.
Cropping/slice preprocessing остаются исходными16x96x96 authorROI.

## Features, окружение и ресурсы

Python3.12.14,Torch2.11.0,torchvision0.26.0,Pillow11.3.0,NumPy2.1.3,sklearn1.6.1.
Отдельный torchvision target под ignored runtime cache; основной venv не меняется.
CPU2threads,seed42,deterministic,eval/inference_mode. Official external weights
скачать один раз в private cache, сохранить full SHA256 и source URL.
Это model weights, не добор пациентов/новые dataset cases.

Grayscale повторить в3channels; resize целой ROI до224x224 bilinear/antialias,
без center crop чтобы не обрезать authorROI; ImageNet mean/std normalization.
Это явное отличие от default torchvision resize256+center224 transform.
Удалить1000class fc (Identity), усреднить512dimfeatures всех16slices каждого bag.
Frozen features могут быть посчитаны сразу для всех TRAIN, т.к. external weights
не зависят от TMJ labels. Feature normalization/scaler остаются fit-fold-only.

Feature extraction<=900s после загрузки weights; supervised fit<=120s,
раздельные статусы и bounds. До inference сохранить program,manifest/config hashes.
Private ignored resnet-features-cv-20261006 хранит weights binding,features,
fold/scaler/head configs,predictions иaggregate report. Нет cloud jobs/paidtraining.

## Оценка и ограничения

Метрики/200paired patient-bootstrap как в предыдущем CV,с одинаковыми draws.
ConvergenceWarning отклоняет run; одноклассовые sourcegroups=null с support.
Это development estimate после ранее просмотренных TRAIN results, не clinical
validation. ImageNet domain/ROI geometry/sourceconfounding ограничения сохраняются.
Предыдущая spatial не выбрана для originalvalidation; новаяcandidate только еслиgo.
Связь FR-ML-OSSEOUS-DEVELOPMENT / FR-ML-OSSEOUS-EVALUATION.

Primary references: [ResNet18 API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
[transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
[Version compatibility](https://github.com/pytorch/vision#installation) подтверждаетTorch2.11/vision0.26.

## Результаты

Запуск завершён успешно: извлечение признаков — 77,67 с, обучение голов,
метрики и bootstrap — 0,60 с; оба ограничения ресурсов выполнены. Все пять
классификаторов сошлись, сохранение и повторная загрузка дали те же предсказания.

| Показатель | Результат |
|---|---:|
| Средний AUROC по пяти фолдам | 0,608655 |
| AUROC объединённых OOF-предсказаний | 0,612921 |
| Average precision объединённых OOF-предсказаний | 0,793153 |
| Средняя разница AUROC относительно spatial | +0,088997 |
| Фолды с улучшением | 4 из 5 |
| AUROC внутри профилей 111 / 151 | 0,639205 / 0,559514 |
| 95% bootstrap-интервал средней разницы, 200 draws | −0,014312 … +0,189600 |

Заранее заданный критерий перехода выполнен. Интервал разницы включает ноль:
результат разрешает следующую проверку, но не доказывает устойчивое превосходство.
Оригинальные validation/test не использованы для inference. Проверены SHA256
фактической программы, manifest и feature cache; review программы не выявил
утечки через scaler или обновление backbone. Перед повторным использованием
runner нужно явно сопоставлять прежние результаты по номеру fold и закрепить
fingerprints прежних report/results/predictions, а не только их manifest.

Program SHA256: `9ee4087429362d2b7c22af3e0729d98b9cb098c299949cdd011d61663631a3e8`.
Manifest SHA256: `d7e6dc97437ee79c755b70bbea449b6f4362b356371f74eff089994dcc127657`.
Артефакты с индивидуальными данными остаются в приватном ignored каталоге;
здесь опубликованы только агрегаты.

## Вывод

Кандидат допущен к следующей проверке. Последующая [original validation](../osseous_resnet_validation_20261006/README.md)
уже выполнена отдельным замороженным протоколом. Модель остаётся
исследовательской; test в этих экспериментах повторно не открывается.
