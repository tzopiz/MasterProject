# Сегментация

Сегментация — отдельное исследовательское направление. Наличие 2D/3D U-Net,
масок или весов не означает подтверждённую точность и не создаёт диагноз.
Численный порог приемлемого качества пока не задан.

<a id="fr-ml-seg-image-mask-alignment"></a>
### FR-ML-SEG-IMAGE-MASK-ALIGNMENT — Согласованность изображения и маски

Объём и маска должны относиться к одной области и иметь согласованные геометрию и преобразования. Пропущенные и несовместимые пары должны учитываться; изменение размера/аугментация применяется согласованно к image и mask.
Приёмка: [TC-ML-SEG-IMAGE-MASK-ALIGNMENT](#tc-ml-seg-image-mask-alignment).

<a id="tc-ml-seg-image-mask-alignment"></a>
### TC-ML-SEG-IMAGE-MASK-ALIGNMENT — проверка

- Предусловие: есть валидная пара, missing mask и несовместимая форма.
- Действие: построить набор и применить преобразование.
- Результат: валидная пара выровнена, исключения доступны для аудита, mask не смещена относительно image.
- Покрытие: Dataset отбирает пары; выделенные тесты сегментации не найдены.

<a id="fr-ml-seg-evaluation-preprocessing"></a>
### FR-ML-SEG-EVALUATION-PREPROCESSING — Совместимость оценки

Оценка checkpoint должна воспроизводить заявленный preprocessing обучения либо явно исследовать и описывать его изменение. Порог бинарной маски и пространственные единицы метрик фиксируются; middle slice не заменяет оценку полного объёма.
Приёмка: [TC-ML-SEG-EVALUATION-PREPROCESSING](#tc-ml-seg-evaluation-preprocessing).

<a id="tc-ml-seg-evaluation-preprocessing"></a>
### TC-ML-SEG-EVALUATION-PREPROCESSING — проверка

- Предусловие: есть checkpoint, учебная нормализация и evaluator.
- Действие: сопоставить преобразования и объёмный отчёт.
- Результат: различия выявлены до вывода; Dice/IoU и дистанции имеют ясную область и единицы.
- Покрытие: Evaluator отличается от train preprocessing. [GAP-ML-SEGMENTATION-PREPROCESSING](../../open-questions/README.md#gap-ml-segmentation-preprocessing).

<a id="fr-ml-seg-mask-evaluation"></a>
### FR-ML-SEG-MASK-EVALUATION — Независимая оценка масок

Вывод о качестве сегментации требует patient-grouped протокола и данных, не использованных для обучения/подбора. Повторное использование train как validation допустимо только как проверка работоспособности без вывода о генерализации.
Приёмка: [TC-ML-SEG-MASK-EVALUATION](#tc-ml-seg-mask-evaluation).

<a id="tc-ml-seg-mask-evaluation"></a>
### TC-ML-SEG-MASK-EVALUATION — проверка

- Предусловие: малый набор вызывает fallback train=val.
- Действие: проверить split и интерпретацию отчёта.
- Результат: совпадение помечено как технический прогон; независимое качество не заявлено.
- Покрытие: Trainers делят volumes/crops, 3D имеет train=val fallback. [GAP-ML-SEGMENTATION-SPLIT](../../open-questions/README.md#gap-ml-segmentation-split).
