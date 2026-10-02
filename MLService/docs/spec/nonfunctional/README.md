# Нефункциональные требования MLService

Применяются общие [конституция](../../../../docs/spec/constitution/README.md) и
[соглашения](../../../../docs/spec/conventions/README.md). Ниже — только ML-специфика.

<a id="nfr-ml-experiment-reproducibility"></a>
### NFR-ML-EXPERIMENT-REPRODUCIBILITY — Воспроизводимость эксперимента

До запуска фиксируются данные/группы, baseline, метрика, выбор модели и порога, ресурсы и место сохранения в протоколе эксперимента. После запуска сохраняются фактические команды, конфигурация, версии входов, состав фолдов и связь отчёта с checkpoint. Неудачный результат также сохраняется.
Приёмка: [TC-ML-NFR-EXPERIMENT-REPRODUCIBILITY](#tc-ml-nfr-experiment-reproducibility).

<a id="tc-ml-nfr-experiment-reproducibility"></a>
### TC-ML-NFR-EXPERIMENT-REPRODUCIBILITY — проверка

- Предусловие: готовится новый эксперимент.
- Действие: сопоставить протокол с сохранёнными артефактами.
- Результат: другой исследователь понимает входы и решения; отсутствующие артефакты названы, а не восстановлены предположением.
- Покрытие: Протокол — [experiments/template](../../../experiments/template/README.md); текущий CV не сохраняет полный паспорт. [GAP-ML-CV-ARTIFACTS](../open-questions/README.md#gap-ml-cv-artifacts).

<a id="nfr-ml-data-privacy"></a>
### NFR-ML-DATA-PRIVACY — Приватность исследовательских материалов

Публичные спецификации, отчёты и коммиты не должны содержать имена пациентов, исходные приватные пути, DICOM с PHI, секреты или приватные манифесты. Псевдонимизация ключа не считается доказательством полной анонимности; provenance публикуется без раскрытия идентичности.
Приёмка: [TC-ML-NFR-DATA-PRIVACY](#tc-ml-nfr-data-privacy).

<a id="tc-ml-nfr-data-privacy"></a>
### TC-ML-NFR-DATA-PRIVACY — проверка

- Предусловие: созданы индекс, cache и отчёт.
- Действие: проверить материалы перед публикацией.
- Результат: приватные соответствия остаются вне Git; публичный отчёт сохраняет воспроизводимую семантику без идентификаторов пациентов.
- Покрытие: Есть stripping/anonymization tools; полнота обезличивания и cache требуют отдельного аудита.

<a id="nfr-ml-upload-boundary"></a>
### NFR-ML-UPLOAD-BOUNDARY — Безопасная граница загрузки

Загруженные файлы должны оставаться внутри временной области запроса и не перезаписывать друг друга из-за имени. Входы проверяются как одна поддерживаемая серия; недопустимый набор завершается явной ошибкой. Ограничения ресурсов определяются до эксплуатационного применения.
Приёмка: [TC-ML-NFR-UPLOAD-BOUNDARY](#tc-ml-nfr-upload-boundary).

<a id="tc-ml-nfr-upload-boundary"></a>
### TC-ML-NFR-UPLOAD-BOUNDARY — проверка

- Предусловие: есть путь с выходом из каталога, duplicate filename и смешанные серии.
- Действие: проверить обработку загрузки.
- Результат: запись вне области невозможна; конфликт и недопустимая серия не становятся successful inference.
- Покрытие: app.py соединяет имя напрямую; ограничения и series validation отсутствуют. [GAP-ML-UPLOAD-VALIDATION](../open-questions/README.md#gap-ml-upload-validation), [OQ-ML-INPUT-RESOURCE-LIMITS](../open-questions/README.md#oq-ml-input-resource-limits).
