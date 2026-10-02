# Приёмка и трассировка iOS

Критерии приёмки находятся рядом с владельцами требований; здесь собраны связи и покрытие.
Статическое чтение не подтверждает запуск на устройстве или прохождение приёмки.
В доступном клиентском коде не обнаружены автоматические тесты этого сценария.
Технические основания: [реализация](../../ai/README.md); известные расхождения: [GAP](../open-questions/README.md#gap).

## Связь требований и оснований

| Требование | Приёмка | Статическое основание; предел доказательства |
|---|---|---|
| [FR-IOS-CHOOSE-SERIES](../functional/analysis/README.md#fr-ios-choose-series) | [TC-IOS-SERIES-SELECTION](../functional/analysis/README.md#tc-ios-series-selection) | Выбор папки в [UI](../../ai/README.md#ui); не проверен запуск |
| [FR-IOS-COMPLETE-SERIES](../functional/analysis/README.md#fr-ios-complete-series) | [TC-IOS-COMPLETE-OPERATION](../functional/analysis/README.md#tc-ios-complete-operation) | [Подготовка](../../ai/README.md#preparation); GAP-IOS-PARTIAL-UPLOAD, поведение при смене выбора не проверено |
| [FR-IOS-TASK-IDENTITY](../functional/analysis/README.md#fr-ios-task-identity) | [TC-IOS-TASK-IDENTITY](../functional/analysis/README.md#tc-ios-task-identity) | [Передача](../../ai/README.md#wire); проверки совпадения задачи не обнаружены |
| [FR-IOS-TERMINAL-OUTCOME](../functional/analysis/README.md#fr-ios-terminal-outcome) | [TC-IOS-TASK-STATES](../functional/analysis/README.md#tc-ios-task-states) | [Ожидание](../../ai/README.md#polling); GAP-IOS-POLLING-ROUTE и GAP-IOS-HIDDEN-PAYLOAD-ERROR |
| [FR-IOS-VALIDATE-LOCALIZATION](../functional/analysis/README.md#fr-ios-validate-localization) | [TC-IOS-INVALID-LOCALIZATION](../functional/analysis/README.md#tc-ios-invalid-localization) | [Преобразование](../../ai/README.md#result); GAP-IOS-HIDDEN-PAYLOAD-ERROR и GAP-IOS-ARRAY-VALIDATION |
| [FR-IOS-PRESENT-LOCALIZATION](../functional/analysis/README.md#fr-ios-present-localization) | [TC-IOS-PARTIAL-RESULT](../functional/analysis/README.md#tc-ios-partial-result) | [Представление](../../ai/README.md#result); GAP-IOS-HIDDEN-PAYLOAD-ERROR |
| [FR-IOS-OPERATION-FEEDBACK](../functional/analysis/README.md#fr-ios-operation-feedback) | [TC-IOS-ERROR-RECOVERY](../functional/analysis/README.md#tc-ios-error-recovery) | [UI](../../ai/README.md#ui) и [сеть](../../ai/README.md#wire); GAP-IOS-ERROR-CONTEXT |
| [NFR-IOS-DATA-BOUNDARY](../nonfunctional/README.md#nfr-ios-data-boundary) | [TC-IOS-DATA-BOUNDARY](../nonfunctional/README.md#tc-ios-data-boundary) | [Выбор и передача](../../ai/README.md#preparation); runtime и приватность сообщений не проверены |
| [NFR-IOS-FILE-ACCESS](../nonfunctional/README.md#nfr-ios-file-access) | [TC-IOS-FILE-ACCESS](../nonfunctional/README.md#tc-ios-file-access) | [Подготовка](../../ai/README.md#preparation); GAP-IOS-PARTIAL-UPLOAD |
| [NFR-IOS-ACCESSIBLE-UI](../nonfunctional/README.md#nfr-ios-accessible-ui) | [TC-IOS-ACCESSIBLE-UI](../nonfunctional/README.md#tc-ios-accessible-ui) | [UI](../../ai/README.md#ui) и [представление](../../ai/README.md#result); GAP-IOS-ARRAY-VALIDATION, доступность не проверена |
