# Трассировка требований Backend

Все требования имеют владельца и приёмочный сценарий в одном документе.
TC — спецификация проверки, а не существующий тест.

| Требование, владелец | Приёмка | Автоматическое покрытие / состояние |
|---|---|---|
| [FR-BE-SERIES-SUBMIT](../functional/analysis/README.md#fr-be-series-submit) | [TC-BE-SERIES-SUBMIT](../functional/analysis/README.md#tc-be-series-submit) | Нет; multipart не проверен |
| [FR-BE-EMPTY-SERIES](../functional/analysis/README.md#fr-be-empty-series) | [TC-BE-EMPTY-SERIES](../functional/analysis/README.md#tc-be-empty-series) | Нет; guard подтверждён кодом |
| [FR-BE-TASK-IDENTITY](../functional/analysis/README.md#fr-be-task-identity) | [TC-BE-TASK-IDENTITY](../functional/analysis/README.md#tc-be-task-identity) | Нет; создание ID подтверждено кодом |
| [FR-BE-TASK-PROGRESS](../functional/analysis/README.md#fr-be-task-progress) | [TC-BE-TASK-PROGRESS](../functional/analysis/README.md#tc-be-task-progress) | Нет; GAP-BE-ML-VALIDATION |
| [FR-BE-RESULT-ACCESS](../functional/analysis/README.md#fr-be-result-access) | [TC-BE-RESULT-ACCESS](../functional/analysis/README.md#tc-be-result-access) | Нет; чтение подтверждено кодом |
| [FR-BE-LOCALIZATION](../functional/analysis/README.md#fr-be-localization) | [TC-BE-LOCALIZATION](../functional/analysis/README.md#tc-be-localization) | Нет; преобразование сверено статически |
| [FR-BE-FAILURE-VISIBLE](../functional/analysis/README.md#fr-be-failure-visible) | [TC-BE-FAILURE-VISIBLE](../functional/analysis/README.md#tc-be-failure-visible) | Нет; GAP-BE-WORKER |
| [FR-BE-TASK-LOOKUP](../functional/analysis/README.md#fr-be-task-lookup) | [TC-BE-TASK-LOOKUP](../functional/analysis/README.md#tc-be-task-lookup) | Нет; отказы подтверждены кодом |
| [NFR-BE-SERIES-INTEGRITY](../nonfunctional/README.md#nfr-be-series-integrity) | [TC-BE-SERIES-INTEGRITY](../nonfunctional/README.md#tc-be-series-integrity) | Нет; GAP-BE-FILE-LOSS |
| [NFR-BE-UPLOAD-BOUND](../nonfunctional/README.md#nfr-be-upload-bound) | [TC-BE-UPLOAD-BOUND](../nonfunctional/README.md#tc-be-upload-bound) | Нет; лимит есть, HTTP не проверен |
| [NFR-BE-HEALTH-SCOPE](../nonfunctional/README.md#nfr-be-health-scope) | [TC-BE-HEALTH-SCOPE](../nonfunctional/README.md#tc-be-health-scope) | Нет; обработчик сверён статически |
| [FR-BE-CLIENT-COMPATIBILITY](../integration/README.md#fr-be-client-compatibility) | [TC-BE-CLIENT-COMPATIBILITY](../integration/README.md#tc-be-client-compatibility) | Нет; GAP-BE-CLIENT-ROUTE, GAP-BE-POLL-WINDOW |
| [FR-BE-ML-CORRELATION](../integration/README.md#fr-be-ml-correlation) | [TC-BE-ML-CORRELATION](../integration/README.md#tc-be-ml-correlation) | Нет; GAP-BE-ML-VALIDATION |

В [Package.swift](../../../Package.swift) только executable target; test target
и отслеживаемые файлы Backend-тестов отсутствуют. [CI](../../../../.github/workflows/ci.yml)
содержит сборку Backend, которая не доказывает выполнение перечисленных TC.
При появлении проверок сюда добавляются точный путь теста, проверяемое поведение
и пределы покрытия. Результаты конкретного запуска остаются в PR.
