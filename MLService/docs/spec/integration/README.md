# Интеграция: вклад MLService

Каноническая граница системы находится в [корневой интеграции](../../../../docs/spec/integration/README.md).
Этот документ задаёт только обязательства ML; маршруты, хранение и состояния
Backend/iOS здесь не определяются. Текущий протокол разобран в [ai/inference](../../ai/inference/README.md).

<a id="fr-ml-int-localization-exchange"></a>
### FR-ML-INT-LOCALIZATION-EXCHANGE — Локализационный обмен

ML должен принимать идентификатор задания и файлы серии, возвращать этот идентификатор, явный исход обработки и результат локализации либо объяснимую ошибку. Успех требует обеих сторон и исходной формы объёма.
Приёмка: [TC-ML-INT-LOCALIZATION-EXCHANGE](#tc-ml-int-localization-exchange).

<a id="tc-ml-int-localization-exchange"></a>
### TC-ML-INT-LOCALIZATION-EXCHANGE — проверка

- Предусловие: переданы корректный task_id и серия; затем модель недоступна.
- Действие: проверить успешный и ошибочный ответ.
- Результат: идентификатор сохранён, success содержит полную локализацию, failure не выдаёт валидный результат.
- Покрытие: Статическая сверка app.py; HTTP-сценарии не выполнены. [Общий обмен](../../../../docs/spec/integration/README.md#int-mp-inference).

<a id="fr-ml-int-result-geometry"></a>
### FR-ML-INT-RESULT-GEOMETRY — Смысл результата

Локализационный результат должен содержать конечные center[3], bbox[6] и volume_shape[3] с согласованными границами в исходном voxel space. Порядок center — z,y,x, bbox — z1,y1,x1,z2,y2,x2. Воксели не представляются миллиметрами без дополнительного согласованного преобразования.
Приёмка: [TC-ML-INT-RESULT-GEOMETRY](#tc-ml-int-result-geometry).

<a id="tc-ml-int-result-geometry"></a>
### TC-ML-INT-RESULT-GEOMETRY — проверка

- Предусловие: получены malformed массивы и граничный центр.
- Действие: проверить DTO и геометрическую согласованность.
- Результат: неполные/неконечные данные отвергнуты; корректный bbox лежит внутри volume_shape.
- Покрытие: Pydantic задаёт List без длины и finite constraints. [GAP-ML-RESULT-VALIDATION](../open-questions/README.md#gap-ml-result-validation).

<a id="fr-ml-int-research-boundary"></a>
### FR-ML-INT-RESEARCH-BOUNDARY — Граница исследовательских возможностей

HTTP должен точно различать доступность сервиса и загруженность модели. Наличие health, весов или локализации не означает поддержку классификации, сегментации, диагноза или клинической пригодности. Подключение нового результата требует отдельного согласованного изменения контракта.
Приёмка: [TC-ML-INT-RESEARCH-BOUNDARY](#tc-ml-int-research-boundary).

<a id="tc-ml-int-research-boundary"></a>
### TC-ML-INT-RESEARCH-BOUNDARY — проверка

- Предусловие: сервис доступен, модель не загружена.
- Действие: проверить readiness и перечень результатов.
- Результат: готовность инференса не выводится из health ok; ответ содержит только поддерживаемые возможности.
- Покрытие: health содержит model_loaded; HTTP использует только регрессию. [Исследовательская граница](../../../../docs/spec/integration/README.md#int-mp-research-boundary).
