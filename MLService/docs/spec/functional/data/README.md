# Данные: идентичность, метки и происхождение

Единицы данных — пациент, исследование/серия и сторона сустава. Они различны:
несколько серий и две стороны пациента не являются независимыми пациентами.
Технический join и форматы разобраны в [ai/data](../../../ai/data/README.md).

<a id="fr-ml-data-patient-group"></a>
### FR-ML-DATA-PATIENT-GROUP — Идентичность пациента

Для оценки все серии и стороны одного пациента должны иметь одну проверенную группу. Идентификатор серии и обезличенный DICOM PatientID не являются доказательством patient identity. Неопределённая связь должна быть отмечена до формирования независимых групп.
Приёмка: [TC-ML-DATA-PATIENT-GROUP](#tc-ml-data-patient-group).

<a id="tc-ml-data-patient-group"></a>
### TC-ML-DATA-PATIENT-GROUP — проверка

- Предусловие: есть несколько серий и сторон с известным общим пациентом.
- Действие: сформировать индекс и разбиение.
- Результат: все связанные записи имеют одну группу; train и итоговая оценка не пересекаются.
- Покрытие: Canonical synthetic tests проверяют supplied source-qualified patient groups, repeat studies и порядок входа; legacy split tests сохранены. Идентичность реальной когорты не проверена. [GAP-ML-PATIENT-IDENTITY](../../open-questions/README.md#gap-ml-patient-identity).

<a id="fr-ml-data-label-join"></a>
### FR-ML-DATA-LABEL-JOIN — Однозначное связывание

Связь серии с разметкой должна быть однозначной и проверяемой. Дубликат ключа, конфликт меток, отсутствующая запись или неполная метка не должны молча создавать достоверный пример. Исключения учитываются с причиной; частичное использование допускается только как явно описанный набор.
Приёмка: [TC-ML-DATA-LABEL-JOIN](#tc-ml-data-label-join).

<a id="tc-ml-data-label-join"></a>
### TC-ML-DATA-LABEL-JOIN — проверка

- Предусловие: есть совпадение, пропуск и два конфликтующих ключа.
- Действие: построить учебный индекс.
- Результат: корректное совпадение сохранено, неоднозначность выявлена, исключения отражены в отчёте.
- Покрытие: Canonical intake отклоняет дубликаты ключей, конфликт patient linkage, неподтверждённую применимость и invalid paths с privacy-safe отчётом; synthetic tests и CLI проверки существуют. Legacy name join сохраняет исторические ограничения. [GAP-ML-LABEL-JOIN-VALIDATION](../../open-questions/README.md#gap-ml-label-join-validation).

<a id="fr-ml-data-label-semantics"></a>
### FR-ML-DATA-LABEL-SEMANTICS — Смысл меток

Сагиттальные и фронтальные метки должны сохранять плоскость и сторону. Коды 1/2/3 означают central/anterior/posterior, 4/5/6 — central/medial/lateral. Код другой плоскости или отсутствие метки не могут означать central. Бинарное central=0/non-central=1 строится только из проверенного класса.
Приёмка: [TC-ML-DATA-LABEL-SEMANTICS](#tc-ml-data-label-semantics).

<a id="tc-ml-data-label-semantics"></a>
### TC-ML-DATA-LABEL-SEMANTICS — проверка

- Предусловие: есть допустимые коды, код иной плоскости и пропуск.
- Действие: проверить mapping и бинаризацию.
- Результат: правильные классы сохранены; ошибочные и отсутствующие значения не преобразованы в норму.
- Покрытие: Canonical schema-v1 и binary conversion проверены synthetic tests на boolean/invalid/wrong-plane/missing codes; absent frontal разрешена только для sagittal-only и не становится normal.

<a id="fr-ml-data-study-label"></a>
### FR-ML-DATA-STUDY-LABEL — Принадлежность метки исследованию

Перенос одной пациентской метки на повторные серии должен иметь подтверждённую применимость к состоянию и времени исследования. До такого подтверждения повторная серия не даёт дополнительного независимо размеченного наблюдения.
Приёмка: [TC-ML-DATA-STUDY-LABEL](#tc-ml-data-study-label).

<a id="tc-ml-data-study-label"></a>
### TC-ML-DATA-STUDY-LABEL — проверка

- Предусловие: у пациента есть повторные исследования и одна запись разметки.
- Действие: проверить основания назначения меток.
- Результат: каждая серия имеет подтверждение применимости либо явное ограничение/исключение.
- Покрытие: Canonical intake требует отдельную confirmed applicability для каждого исследования; synthetic test отклоняет unknown. Реальную применимость должен подтвердить владелец; legacy join распространяет одну метку на совпавшие имена. [OQ-ML-REPEATED-STUDY-LABEL](../../open-questions/README.md#oq-ml-repeated-study-label).

<a id="fr-ml-data-data-provenance"></a>
### FR-ML-DATA-DATA-PROVENANCE — Паспорт производных данных

Используемый кроп или нормализованный объём должен быть связан с исходной серией, версией разметки, преобразованием и моделью локализации. Состав групп и исключений фиксируется до сравнения моделей; существующий файл сам по себе не подтверждает совместимость версии.
Приёмка: [TC-ML-DATA-DATA-PROVENANCE](#tc-ml-data-data-provenance).

<a id="tc-ml-data-data-provenance"></a>
### TC-ML-DATA-DATA-PROVENANCE — проверка

- Предусловие: подготовлены две версии ROI или preprocessing.
- Действие: проверить паспорт входов эксперимента.
- Результат: версии различимы, файлы и группы прослеживаются, несовместимые производные не смешиваются.
- Покрытие: Metadata кропов существует; полный fingerprint и контроль cache не обеспечены. [GAP-ML-DERIVED-DATA-PROVENANCE](../../open-questions/README.md#gap-ml-derived-data-provenance).

## Канонический исследовательский вход

Schema-v1 studies/labels, API и CLI определены в [инженерном контракте данных](../../../ai/data/README.md).
Основной путь использует supplied source/patient/study/label-record IDs и
явную применимость метки к конкретной study-row; неизвестные связи блокируют
зависимый этап. Names/series/DICOM identifiers не восстанавливают patient identity.
source_id остаётся одним пространством идентичности между releases одной когорты;
известные повторные пациенты в нескольких источниках требуют проверенного mapping
до объединения. Legacy name join не подтверждает выполнение этих правил.

Наличие пути кропа завершает только intake stage; подтверждение training readiness
требует паспорта ROI. Подготовка uncropped входа не объявляется готовым обучением.
