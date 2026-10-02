# Интеграция iOS с Backend

Каноническое парное соглашение находится в [общем интеграционном контракте](../../../../docs/spec/integration/README.md).
Этот раздел задаёт вклад iOS-клиента; он не дублирует внутреннее устройство Backend или ML Service.
Текущие адреса, маршруты, структуры ответов и преобразования описаны в [реализации](../../ai/README.md).

## Передача серии

Клиент отвечает за осознанный выбор серии, доступность всех выбранных файлов и явный запуск передачи.
Способ упаковки, имена частей и ответ принятия определяются общим контрактом.
Клиент должен различать ошибку передачи и отказ принять запрос.
Проверка по клиентским признакам не доказывает медицинскую или геометрическую корректность DICOM-серии.

Связанные требования: [FR-IOS-COMPLETE-SERIES](../functional/analysis/README.md#fr-ios-complete-series), [FR-IOS-TASK-IDENTITY](../functional/analysis/README.md#fr-ios-task-identity), [NFR-IOS-FILE-ACCESS](../nonfunctional/README.md#nfr-ios-file-access).

## Состояние задачи

Идентификатор принятой задачи принадлежит Backend и используется клиентом без подмены локальным идентификатором запроса.
Клиент получает состояние через существующий согласованный маршрут.
Он не предполагает наличие отдельного маршрута состояния, если такого маршрута нет в парном соглашении.
Нетёрминальные, успешное и ошибочное состояния интерпретируются по общему контракту.
Частота опроса и общий бюджет ожидания остаются предметом [OQ-IOS-WAITING-POLICY](../open-questions/README.md#oq-ios-waiting-policy).

Связанные требования: [FR-IOS-TASK-IDENTITY](../functional/analysis/README.md#fr-ios-task-identity), [FR-IOS-TERMINAL-OUTCOME](../functional/analysis/README.md#fr-ios-terminal-outcome), [FR-IOS-OPERATION-FEEDBACK](../functional/analysis/README.md#fr-ios-operation-feedback).

## Результат и ошибка

Backend — источник состояния задачи и предоставленного результата; клиент отвечает за безопасную интерпретацию и отображение.
Клиент проверяет идентичность задачи, терминальное состояние, структуру и допустимость предоставленных координат.
Отсутствующее поле отличается от поля, которое присутствует, но не может быть разобрано.
Числовые координаты должны сохранять оси, единицы и сторону, установленные общим контрактом.
Отсутствие стороны не превращается в отрицательный медицинский вывод.
Если Backend сообщил причину ошибки, клиент должен сохранить её смысл при отображении.

Связанные требования: [FR-IOS-VALIDATE-LOCALIZATION](../functional/analysis/README.md#fr-ios-validate-localization), [FR-IOS-PRESENT-LOCALIZATION](../functional/analysis/README.md#fr-ios-present-localization), [FR-IOS-OPERATION-FEEDBACK](../functional/analysis/README.md#fr-ios-operation-feedback).

## Среда и приватность

Адрес Backend должен соответствовать фактической среде клиента; выбор механизма конфигурации не согласован: [OQ-IOS-BACKEND-ADDRESS](../open-questions/README.md#oq-ios-backend-address).
Условия защищённой передачи и допуска к среде задаются на общей границе системы: [OQ-IOS-MEDICAL-DATA](../open-questions/README.md#oq-ios-medical-data).
Клиент не должен самостоятельно отправлять DICOM в ML Service или посторонние системы.

Связанное требование: [NFR-IOS-DATA-BOUNDARY](../nonfunctional/README.md#nfr-ios-data-boundary).
