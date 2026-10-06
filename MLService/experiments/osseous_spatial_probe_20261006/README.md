# Osseous spatial-head TRAIN-only probe — 2026-10-06

Статус: completed

## Вопрос и критерий решения

Проверить, позволяет ли пространственная голова той же CNN выучить небольшой
TRAIN subset. Исходная глобально усредняющая модель не достигла fit-критерия.
Candidate сохраняет conv filters/pool/slice mean, заменяет global average pool
на adaptive 4x4 и голову Linear(256,32), ReLU, Linear(32,1). Это единая
architecture гипотеза о сохранении пространственного контекста; нельзя отдельно
приписать эффект pooling или дополнительной ёмкости головы.

Обе модели переобучаются с нуля на одинаковом subset и в одной среде.
Fit критерий: AUROC >=.95 и final weighted BCE <=50% initial BCE. Предпочесть
candidate для следующего development comparison только если критерий выполнен,
а исходный baseline на том же budget его не достиг. Иначе пересмотреть гипотезу.
Связь: FR-ML-OSSEOUS-DEVELOPMENT, FR-ML-OSSEOUS-EVALUATION.

## Данные и разбиение

Повторить выбранные 16 TRAIN пациентов / 32 стороны из private report первого
train-diagnostic-20261006; сверить IDs с frozen TRAIN membership и их checksum.
Ни validation, ни test crop/prediction не открываются. Test membership только
проверяется на отсутствие пересечения. Split digest:
c65cb2ef35e42b042b4607d33dc0c387b57021ea8309c6cee42eab1d38a418bc.
Author ROI, те же 16x96x96 NPZ и p1-p99/square resize; новая загрузка запрещена.

## Воспроизведение и ресурсы

Перед запуском проверены Python 3.12.14, Torch 2.11.0, NumPy 2.1.3,
sklearn 1.6.1. CPU, 2 threads, seed42, deterministic algorithms.
На каждую модель: 40 эпох, batch4, AdamW lr .001/default weight decay,
TRAIN-only pos_weight, без augmentation/early stopping. Общий wall budget180s.
Приватная программа, её hash, trainer snapshot, selected IDs и epoch history:
ignored research-20261005/spatial-diagnostic-20261006. Новая paid job не нужна.

## Оценка

Сравнение initial/final TRAIN weighted BCE, AUROC, probability std, gradient norm.
Нет подбора по validation/test; fit не подтверждает generalization. Прямого
сравнения с cloud Torch2.6 checkpoint не проводится. Нет clinical claim.
Если выбран candidate, следующий этап — отдельный patient-grouped development
протокол и проверка исходного source-profile confounding.

## Результаты

Обе модели завершили по 40 эпох, суммарно 26.99 секунды CPU.

| Модель | Parameters | Initial/final BCE | Final AUROC | Probability std |
|---|---:|---:|---:|---:|
| global baseline | 1265 | .69727 / .69048 | .62109 | .00594 |
| spatial head | 9505 | .69444 / .62470 | .79297 | .07416 |

Ни одна не достигла fit-критерия. Для candidate reduction BCE около10%, не50%.
Private report SHA256 программы:
152b4d2182fc8831b60c935668ed512340aaf880c33113cd59aff4d1b1e5b4f3.
Исходный subset digest:
d5a730595855d7465f74cca0be962f6825ff32cdf3aec94832c979d71542dc85.
Связанный index SHA256:
f2aa5b44e3931dfe88a6c16f08c12cd916ba58256766c5b39da02709e728f235.
Новых validation/test inference и платных jobs нет.

## Вывод

Architecture fit гипотеза при 40 эпохах не подтверждена; candidate не выбирается
для quality comparison по частичному улучшению. Spatial head даёт более широкий
разброс probabilities и более низкий TRAIN loss, но остаётся слабой на subset.
Каждая эпоха содержит всего8 optimizer steps: 320 за run, тогда как 40 эпох полного
TRAIN содержат2040 steps. Следующая отдельная гипотеза — ограниченный step budget.
