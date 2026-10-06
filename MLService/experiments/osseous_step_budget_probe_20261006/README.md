# Osseous TRAIN-only step-budget probe — 2026-10-06

Статус: completed

## Вопрос и критерий решения

40 subset epochs дают320 updates, полный TRAIN40epochs —2040. Проверить,
позволит ли увеличенный budget1600updates выучить subset без иных изменений.
Повторить обе модели из spatial probe с нуля, изменив только epochs40→200.
Fit: final TRAIN AUROC>=.95 и weighted BCE<=50% initial.
Если только spatial проходит, направить её на отдельный patient-grouped
TRAIN CV при стандартном full-data budget<=40epochs. Если обе не проходят,
не тратить GPU на эту архитектуру; исследовать признаки/pretraining.

## Данные и границы

Те же16TRAINпациентов/32стороны и checksum из предыдущего spatial probe.
Источник TMJ-OD3D V2 selected author ROI. Test/validation pixels/predictions
не читаются. Split/subset/index SHA256 фиксированы предыдущим протоколом.
FR-ML-OSSEOUS-DEVELOPMENT, FR-ML-OSSEOUS-EVALUATION.

## Ресурсы и воспроизведение

Python3.12.14,Torch2.11.0,NumPy2.1.3,CPU2threads,seed42,deterministic.
Global baseline1265parameters и spatial-head9505parameters. AdamW .001,
batch4,TRAIN-only weighted BCE,без augmentation/early stopping.
200epochs каждой модели, общий wall180s. Это локальная capacity диагностика
с новым budget; обычный trainer сохраняет свой предел40epochs и не запускается.
Программа/history/selectedIDs/hash только ignored spatial-step-diagnostic-20261006.
Никаких cloud jobs/расходов и новых dataset downloads.

## Оценка и ограничения

Вычислять TRAIN loss/AUROC,probability spread,gradient norm. Fit на обучающих
примерах не доказывает generalization и не является улучшением final test.
Другие признаки/learning rate/данные одновременно не менять.

## Результаты

Обе модели завершили200epochs/1600updates, общий CPU wall136.96s.

| Модель | Initial/final BCE | TRAIN AUROC | Fit criterion |
|---|---:|---:|---|
| global baseline | .69727 / .67394 | .70313 | failed |
| spatial head | .69444 / .13738 | 1.00000 | passed |

Фактический Torch2.11.0; source/index/subset fingerprints совпали с spatial probe.
Private script SHA256:
ca2dccd731cdd47769a8235fe7a1130c3552e7d42fc8fd6de26cb77a3ea9f7a5.
Ни validation/test, ни cloud jobs не использовались.

## Вывод

Только spatial-head достигла заранее заданного fit-критерия.
Переход к patient-grouped TRAIN CV на стандартном40epoch budget разрешён
capacity gate. Это улучшение memorization, не подтверждённое generalization.
