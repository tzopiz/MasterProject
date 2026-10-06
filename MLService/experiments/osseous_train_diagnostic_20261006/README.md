# Osseous TRAIN-only overfit diagnostic — 2026-10-06

Статус: completed

## Вопрос и критерий решения

Способна ли исходная CNN выучить небольшую тренировочную выборку при неизменном
loss/optimizer? Это диагностика оптимизации/ёмкости, не quality experiment.
Признак успешного fit: TRAIN AUROC >=0.95 и weighted BCE уменьшается >=50%.
Неуспех за ограниченное время не доказывает отсутствие медицинского сигнала.
Связь: FR-ML-OSSEOUS-DEVELOPMENT, FR-ML-OSSEOUS-EVALUATION; change 011.

## Данные и разбиение

Только уже закреплённый TRAIN выбранной TMJ-OD3D V2 когорты. Выбрать первые
8 bilateral-normal и 8 bilateral-pathology пациентов в sorted pseudonymous
порядке; 16 пациентов / 32 стороны. Discordant стороны исключены только из
этого диагностического subset. Ни validation, ни test pixels/предсказания
не читаются. Frozen split digest:
c65cb2ef35e42b042b4607d33dc0c387b57021ea8309c6cee42eab1d38a418bc.
Общая cohort: 146 пациентов / 289 сторон. Источник archive-prefix; не random sample.

## Воспроизведение и ресурсы

Локальный CPU, pinned Python 3.12 ML venv, torch 2.6.0, seed 42,
16x96x96 author bags, существующие p1-p99 normalization/square resize.
Исходная SliceBagClassifier (1265 parameters), weighted BCE (subset TRAIN only),
AdamW lr=.001, batch4, <=40 эпох, wall time <=180s, без early stopping/augmentation.
Точная приватная программа/конфигурация, исходный trainer hash и report сохраняются
в ignored research-20261005/train-diagnostic-20261006; новая оплачиваемая job не нужна.
Программа фиксирует loss/AUROC/probability std/gradient norm по эпохам; weights не публикуются.

## Оценка и ограничения

Метрики оцениваются на том же subset, на котором обучались: memorization
не доказывает generalization. Не выбирается кандидат по final test. Дополнительный
baseline p=0.5 и его weighted BCE. Никакая оценка произвольного DICOM не заявляется.

## Результаты

Выполнено: 16 TRAIN пациентов / 32 стороны, 40 эпох, 12.87 секунды CPU.
Фактический torch **2.11.0**, вместо запланированного 2.6.0: локальный venv
не совпал с cloud pins. Это отклонение протокола; прямое численное сравнение с
GPU checkpoint 2.6.0 не проводится. Данные validation/test не читались.

| Показатель | До обучения | После 40 эпох |
|---|---:|---:|
| TRAIN subset AUROC | 0.6094 | 0.6211 |
| Weighted BCE | 0.69727 | 0.69048 |
| Probability std | 0.00204 | 0.00594 |

Mean gradient norm последней эпохи 0.2766: градиенты присутствуют.
Критерий fit не достигнут. Private report/program сохраняют выбранные IDs,
всю историю и fingerprints. SHA256 программы:
1e013d9b528ee4d63932d8d7f3a3dc288af21a4a02b5ea2bdae8a54073546848;
trainer snapshot:
83d9803c0c39c251bbda3b18219619970d5f78be08643ea301ced2f33d960d2b.
Команда: local ML venv Python + `train-diagnostic-20261006/probe.private.py`.
Скрипт не сохраняет веса, агрегаты оставлены здесь.

## Вывод

В ограниченном диагностическом прогоне исходная модель не выучила subset.
Это поддерживает проверку ёмкости/пространственных признаков и числа updates,
но не доказывает причины cloud failure или отсутствие медицинского сигнала.
Следующий шаг: отдельный TRAIN-only spatial-model probe, затем patient-grouped
сравнение на development данных. Frozen test сохраняется без нового inference.
