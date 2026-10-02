# Классификация положения головок

Многоклассовая задача различает три положения отдельно для каждой плоскости и
стороны; бинарная объединяет только два класса смещения в non-central.
Исследовательская классификация пока не входит в HTTP-результат анализа.

<a id="fr-ml-cls-task-definition"></a>
### FR-ML-CLS-TASK-DEFINITION — Явная постановка

Эксперимент должен указать вход (весь объём или ROI), плоскости, стороны, классы и преобразование logits в решение. Сагиттальный эксперимент не подтверждает обученность фронтального выхода только потому, что модель содержит две головы.
Приёмка: [TC-ML-CLS-TASK-DEFINITION](#tc-ml-cls-task-definition).

<a id="tc-ml-cls-task-definition"></a>
### TC-ML-CLS-TASK-DEFINITION — проверка

- Предусловие: модель содержит несколько выходов, loss использует один.
- Действие: сверить протокол и отчёт.
- Результат: результат относится только к обученным и оценённым задачам.
- Покрытие: Модели и CV сверены статически; существуют тесты shapes, не качества.

<a id="fr-ml-cls-group-independence"></a>
### FR-ML-CLS-GROUP-INDEPENDENCE — Групповая независимость

Все сравниваемые классификаторы должны использовать зафиксированные группы пациентов. При ROI от обученного детектора необходимо учитывать его учебные группы. Отсутствие пересечения сторон или серий не заменяет отсутствие пересечения пациентов.
Приёмка: [TC-ML-CLS-GROUP-INDEPENDENCE](#tc-ml-cls-group-independence).

<a id="tc-ml-cls-group-independence"></a>
### TC-ML-CLS-GROUP-INDEPENDENCE — проверка

- Предусловие: есть несколько моделей и сведения о detector train.
- Действие: проверить принадлежность всех звеньев к группам.
- Результат: сравнение использует одинаковые группы; утечка через детектор отсутствует или явно ограничивает вывод.
- Покрытие: Тест StratifiedGroupKFold существует; состав detector train требует аудита. [GAP-ML-DETECTOR-LEAKAGE](../../open-questions/README.md#gap-ml-detector-leakage).

<a id="fr-ml-cls-model-selection"></a>
### FR-ML-CLS-MODEL-SELECTION — Отделение подбора от итоговой оценки

Выбор эпохи, гиперпараметров и порога должен быть отделён от данных, на которых заявляется итоговое независимое качество. CV с выбором эпохи по validation того же фолда публикуется как оценка разработки, а не final test.
Приёмка: [TC-ML-CLS-MODEL-SELECTION](#tc-ml-cls-model-selection).

<a id="tc-ml-cls-model-selection"></a>
### TC-ML-CLS-MODEL-SELECTION — проверка

- Предусловие: известны данные выбора checkpoint и вычисления метрик.
- Действие: проверить протокол и формулировку вывода.
- Результат: источник каждого решения указан; независимый результат заявлен только при отделённой оценке.
- Покрытие: Текущая CV выбирает эпоху по тому же val AUC. [GAP-ML-VALIDATION-SELECTION](../../open-questions/README.md#gap-ml-validation-selection).

<a id="fr-ml-cls-threshold-calibration"></a>
### FR-ML-CLS-THRESHOLD-CALIBRATION — Калибровка порога

Порог должен иметь зафиксированный метод и источник калибровки без обращения к итоговым меткам. Для воспроизводимого порога оценочные преобразования входа должны быть детерминированными; train augmentation не считается отключённой вызовом model.eval().
Приёмка: [TC-ML-CLS-THRESHOLD-CALIBRATION](#tc-ml-cls-threshold-calibration).

<a id="tc-ml-cls-threshold-calibration"></a>
### TC-ML-CLS-THRESHOLD-CALIBRATION — проверка

- Предусловие: заданы calibration records и обучающий dataset с augmentation.
- Действие: повторить получение calibration predictions в оценочном режиме.
- Результат: входы воспроизводимы; порог и его происхождение сохранены; итоговая выборка не участвовала.
- Покрытие: CV использует augmented train loader; отдельной проверки источника нет. [GAP-ML-AUGMENTED-CALIBRATION](../../open-questions/README.md#gap-ml-augmented-calibration).

<a id="fr-ml-cls-classification-report"></a>
### FR-ML-CLS-CLASSIFICATION-REPORT — Полный отчёт классификации

Отчёт должен различать плоскости, support классов, базовое сравнение и ошибки. Для бинарной задачи нужны AUC, accuracy, balanced accuracy, sensitivity/specificity, F1 с явным положительным классом и confusion matrix; для трёх классов — macro-F1, balanced accuracy и recall каждого класса. Коллапс в большинство нельзя скрывать средней accuracy.
Приёмка: [TC-ML-CLS-CLASSIFICATION-REPORT](#tc-ml-cls-classification-report).

<a id="tc-ml-cls-classification-report"></a>
### TC-ML-CLS-CLASSIFICATION-REPORT — проверка

- Предусловие: есть несбалансированные labels и константный классификатор.
- Действие: сформировать отчёт рядом с baseline.
- Результат: одноклассовые случаи и нераспознанные классы видны; метрики имеют определённую семантику.
- Покрытие: Binary metrics покрыты существующими тестами; multiclass trainer сообщает преимущественно accuracy. [GAP-ML-INCOMPLETE-CLASS-REPORT](../../open-questions/README.md#gap-ml-incomplete-class-report).

<a id="fr-ml-cls-scientific-conclusion"></a>
### FR-ML-CLS-SCIENTIFIC-CONCLUSION — Сила научного вывода

Сводка должна показывать все запланированные фолды, способ агрегации, неопределённость и ограничения. Стандартное отклонение фолдов не называется доверительным интервалом. Достижение численной цели возможно только после выбора метрики, протокола и порога владельцем исследования.
Приёмка: [TC-ML-CLS-SCIENTIFIC-CONCLUSION](#tc-ml-cls-scientific-conclusion).

<a id="tc-ml-cls-scientific-conclusion"></a>
### TC-ML-CLS-SCIENTIFIC-CONCLUSION — проверка

- Предусловие: есть неполный прогон, среднее/std и заявленная цель.
- Действие: проверить итоговый вывод.
- Результат: неполнота отмечена, std назван корректно, несогласованная цель остаётся открытой.
- Покрытие: JSON содержит completion и std; целевой критерий не утверждён. [OQ-ML-QUALITY-CRITERION](../../open-questions/README.md#oq-ml-quality-criterion).
