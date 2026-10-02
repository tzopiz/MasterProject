# iOS: устройство реализации и основания оценки

Здесь описан текущий код. Желаемое поведение задано в [спецификации](../spec/README.md).
Оценка основана на статическом чтении; сборка, запуск, HTTP и приёмка на устройстве не подтверждены.
В дереве iOS нет обнаруженных файлов автоматических тестов анализа; mock-сеть всегда выбрасывает ошибку и не доказывает прохождение сценария.

<a id="ui"></a>
## Точка входа и UI

[MasterDoctorApp](../../MasterDoctor/MasterDoctor/MasterDoctorApp.swift) открывает [TMJDetectionView](../../MasterDoctor/MainFeatures/Sources/AnalyticsApp/Views/TMJDetectionView.swift) с реальной сетью и JSON decoder.
Экран хранит выбранные URL, результат, сообщение и признак выполнения.
Он выбирает одну папку, фильтрует непосредственные элементы по расширению `.dcm` без учёта регистра и показывает число/раскрываемые имена.
При запуске очищает результат и показывает `Uploading…`; после возврата — `Analysis complete`, после исключения — `Failed: <localizedDescription>`.
Кнопка запуска исчезает во время операции; выбор папки через toolbar остаётся доступным.

<a id="preparation"></a>
## Подготовка файлов

Importer получает security-scoped доступ к папке на время перечисления, затем освобождает его.
[TMJDetectionFetchService](../../MasterDoctor/MainFeatures/Sources/AnalyticsApp/Services/TMJDetectionFetchService.swift) позднее получает доступ к каждому дочернему URL и читает `Data`.
При отказе доступа URL пропускается; если ничего не осталось, возникает `emptySelectedFiles`.
Чтение одного доступного файла может завершить операцию ошибкой; успешный доступ освобождается через `defer`.
Все файлы и multipart собираются в памяти; клиентского лимита размера и числа файлов нет.
Полнота серии и реальное действие разрешений после закрытия доступа к папке не подтверждены.

<a id="wire"></a>
## HTTP и декодирование

[AnalysisEndpoint](../../MasterDoctor/MainFeatures/Sources/AnalyticsApp/Services/Endpoint/AnalysisEndpoint.swift) задаёт базу `http://localhost:8080/api`.
Загрузка серии — `POST /analysis`; запрос состояния — `GET /analysis/{UUID}/status`; результат — `GET /analysis/{UUID}` относительно базы.
[Endpoint](../../MasterDoctor/CommonCore/Sources/CoreNetwork/NetworkService/Interface/Endpoint.swift) формирует повторяющиеся multipart-части `files`, исходные имена, `application/dicom` и boundary из переданного UUID.
[NetworkingService](../../MasterDoctor/CommonCore/Sources/CoreNetwork/NetworkService/NetworkingService.swift) использует `URLSession`, принимает 200–299 и декодирует тело.
Другой код ответа превращается в `URLError(.badServerResponse)` без сохранения кода и тела; отдельного retry нет.
[JSONDecoderService](../../MasterDoctor/Foundation/Sources/FoundationInternal/Converters/BaseDecoder.swift) применяет `convertFromSnakeCase`; nonthrowing overload использует `try?`.
[DTO](../../MasterDoctor/MainFeatures/Sources/AnalyticsApp/Models/AnalyticsResponse.swift): upload ожидает `task_id` UUID; status — UUID, строковый `status`, optional `error_message`.
Raw result содержит UUID, строковый `status`, optional JSON-строки `tmj_left/right`, optional `volume_shape` и даты-строки.
Legacy-поля slices/masks/parameters/diagnosis представлены optional JSON-строками; это не подтверждает их поддержку Backend.

<a id="polling"></a>
## Ожидание

Actor делает до десяти запросов состояния, между нетёрминальными ответами ждёт две секунды, включая последнюю попытку.
`completed` вызывает отдельное получение raw result; `failed` выбрасывает ошибку с optional причиной; любая другая строка продолжает цикл.
Затем возникает `timeout`; это сумма ожиданий плюс HTTP, а не общий deadline.
Возвращённое итоговое состояние и совпадение идентификаторов отдельно не проверяются.
Маршрут `/status` расходится с текущим Backend; нормативное исправление связано с [GAP-IOS-POLLING-ROUTE](../spec/open-questions/README.md#gap-ios-polling-route).

<a id="result"></a>
## Преобразование и представление

Actor nonthrowing-декодированием превращает отсутствующие и некорректные вложенные строки в `nil`.
[BoundingBox](../../MasterDoctor/MainFeatures/Sources/AnalyticsApp/Models/BoundingBox.swift) хранит center `[z,y,x]` и bbox `[z1,y1,x1,z2,y2,x2]`.
[AnalysisResult](../../MasterDoctor/MainFeatures/Sources/AnalyticsApp/Models/AnalysisResult.swift) хранит optional стороны и размер объёма; legacy структуры не означают клиническую функцию.
[ResultsView](../../MasterDoctor/MainFeatures/Sources/AnalyticsApp/Models/AnalysisResult+View.swift) показывает размер и присутствующие стороны численно, без DICOM-изображения.
Views индексируют массивы длиной 3/3/6 без проверок; отсутствующая сторона просто не отображается.
Главный экран не использует отдельные DICOM viewer-компоненты, не рисует область поверх снимка и не представляет классификацию, сегментацию или диагноз.
Эти свойства кода объясняют [GAP-IOS-HIDDEN-PAYLOAD-ERROR](../spec/open-questions/README.md#gap-ios-hidden-payload-error) и [GAP-IOS-ARRAY-VALIDATION](../spec/open-questions/README.md#gap-ios-array-validation), не задают норматив для будущей реализации.

## Работа с документацией

При изменении поведения сначала сверяются требования и относящиеся к ним исходники.
Связи с критериями и достаточность доказательств обновляются в [трассировке](../spec/traceability/README.md).
Приёмка на устройстве, проверка JSON и HTTP должны отмечаться как выполненные только при наличии фактического результата.
Нормативные решения после реализации оформляются по [правилам патчей](../spec/patches/README.md); текущая реализация не может молча заменить требование.
