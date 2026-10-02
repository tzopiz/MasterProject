# Открытые вопросы и известные пробелы iOS

GAP — известное несоответствие действующему требованию; OQ — ещё не согласованное решение.
Основания статической оценки и пути исходников находятся в [документации реализации](../../ai/README.md).

## GAP

<a id="gap-ios-polling-route"></a>
### GAP-IOS-POLLING-ROUTE — Маршрут ожидания

Клиент опрашивает отдельный маршрут состояния, отсутствующий в текущем API Backend.
Это препятствует получению результата после принятой загрузки и нарушает [FR-IOS-TERMINAL-OUTCOME](../functional/analysis/README.md#fr-ios-terminal-outcome).
Закрытие требует согласованного обращения и прохождения [TC-IOS-TASK-STATES](../functional/analysis/README.md#tc-ios-task-states).

<a id="gap-ios-partial-upload"></a>
### GAP-IOS-PARTIAL-UPLOAD — Неполная серия при отказе доступа

После выбора папки её доступ освобождается; при последующем чтении файлы с отказом отдельного доступа пропускаются.
Передача оставшейся части нарушает [FR-IOS-COMPLETE-SERIES](../functional/analysis/README.md#fr-ios-complete-series) и [NFR-IOS-FILE-ACCESS](../nonfunctional/README.md#nfr-ios-file-access).
Закрытие требует [TC-IOS-COMPLETE-OPERATION](../functional/analysis/README.md#tc-ios-complete-operation) и [TC-IOS-FILE-ACCESS](../nonfunctional/README.md#tc-ios-file-access), включая проверку реального файлового доступа.

<a id="gap-ios-hidden-payload-error"></a>
### GAP-IOS-HIDDEN-PAYLOAD-ERROR — Скрытые ошибки вложенных данных

Ошибка разбора блока локализации становится отсутствующим блоком, а итоговое состояние ответа отдельно не проверяется.
Пустые блоки исчезают из результата без объяснения пользователю.
Это расходится с [FR-IOS-TERMINAL-OUTCOME](../functional/analysis/README.md#fr-ios-terminal-outcome), [FR-IOS-VALIDATE-LOCALIZATION](../functional/analysis/README.md#fr-ios-validate-localization) и [FR-IOS-PRESENT-LOCALIZATION](../functional/analysis/README.md#fr-ios-present-localization).
Закрытие требует [TC-IOS-TASK-STATES](../functional/analysis/README.md#tc-ios-task-states), [TC-IOS-INVALID-LOCALIZATION](../functional/analysis/README.md#tc-ios-invalid-localization) и [TC-IOS-PARTIAL-RESULT](../functional/analysis/README.md#tc-ios-partial-result).

<a id="gap-ios-array-validation"></a>
### GAP-IOS-ARRAY-VALIDATION — Непроверенные координатные массивы

Представление обращается к элементам размера объёма, центра и области без проверки длины.
Это нарушает [FR-IOS-VALIDATE-LOCALIZATION](../functional/analysis/README.md#fr-ios-validate-localization) и безопасное отображение из [NFR-IOS-ACCESSIBLE-UI](../nonfunctional/README.md#nfr-ios-accessible-ui).
Закрытие требует [TC-IOS-INVALID-LOCALIZATION](../functional/analysis/README.md#tc-ios-invalid-localization) и [TC-IOS-ACCESSIBLE-UI](../nonfunctional/README.md#tc-ios-accessible-ui).

<a id="gap-ios-error-context"></a>
### GAP-IOS-ERROR-CONTEXT — Неполная передача смысла ошибки

Сетевой слой теряет код и тело ответа при отказе; сообщения этапов сервиса не передаются экрану.
Это оставляет [FR-IOS-OPERATION-FEEDBACK](../functional/analysis/README.md#fr-ios-operation-feedback) неподтверждённым; приёмка — [TC-IOS-ERROR-RECOVERY](../functional/analysis/README.md#tc-ios-error-recovery).

## OQ

<a id="oq-ios-waiting-policy"></a>
### OQ-IOS-WAITING-POLICY — Политика ожидания

Нужны согласованные частота опроса, бюджет времени и пользовательский исход при его исчерпании.
Текущий предел клиента не является соглашением о длительности серверной обработки.
Решение должно уточнить FR-IOS-TERMINAL-OUTCOME и TC-IOS-TASK-STATES вместе с общей интеграцией; до него численный норматив не задан.

Ответ: ожидается решение владельца.

<a id="oq-ios-backend-address"></a>
### OQ-IOS-BACKEND-ADDRESS — Адрес Backend

Не выбраны поддерживаемые среды и способ задания адреса Backend для клиента.
Текущий локальный адрес не доказывает доступность Backend с физического устройства.
Решение должно установить выбор адреса и приёмку доступности, не закрепляя случайное окружение как требование.

Ответ: ожидается решение владельца.

<a id="oq-ios-series-budget"></a>
### OQ-IOS-SERIES-BUDGET — Бюджет серии

Не установлены допустимые число файлов, суммарный размер, потребление памяти и реакция до передачи слишком большой серии.
Текущее формирование всей передачи в памяти не является допустимым бюджетом.

Ответ: ожидается решение владельца.

<a id="oq-ios-medical-data"></a>
### OQ-IOS-MEDICAL-DATA — Условия обработки медицинских данных

Нужно определить поддерживаемую среду, защиту соединения, допуск пользователя и условия хранения переданных DICOM.
Этот вопрос требует общего решения на границе системы; клиентский контракт не обещает отсутствующую аутентификацию или обезличивание.
До решения действует ограничение выбранной серии и назначенного получателя из NFR-IOS-DATA-BOUNDARY.

Ответ: ожидается решение владельца.
