# 008 — Проверенная техническая курация полной когорты

Статус: done. Зависит от 005; нужна при фактических source failures.

## Наблюдение и решение

Полный проход обнаружил annotation_side_conflict. Адресная проверка исходных
L/R JSON подтвердила: filename L содержит shape с R, filename R — shape с L;
авторские коды одинаковые 1,5. Достоверность стороны при противоречии не
восстанавливается догадкой. Такой пациент исключается целиком с причиной.
Это техническая проверка согласованности источника, не новая медицинская разметка.

## План и приёмка

1. Отдельный tools/curate_tmj_od3d.py принимает завершённый source root,
   приватную policy с source preparation SHA256 и точным списком reviewed
   failed patient receipts (hash/code/disposition=exclude/reason).
2. Требовать source complete=true, EOF/1043 coverage, исходный metadata checksum,
   receipt/index согласованность и точное покрытие каждого failure policy.
   Без reviews или при неизвестной причине отказать; не менять live source state.
3. Создать новый приватный immutable-ready индекс и скопировать только проверенные
   crops в отдельный output root (no overlap/no overwrite,≤5GiB). Проверить
   crop checksums и исходные bytes до/после. Исходные failures сохранить в
   source_failures и provenance; failures нового принятого индекса пусты только
   после явного решения. В payload не включать raw processed patient IDs.
4. Приёмка: incomplete source, несовпадение checksum/policy/receipt, пропущенная
   review, изменение crop и overlap отклоняются. Реальная курация выполняется
   после завершения source scan; агрегатный отчёт counts/reasons сохраняется
   без patient rows. Consumer preflight binary/multilabel обязателен отдельно.
5. Целевая реализация компактная, без нового downloader или клинических догадок.
   Политика не допускает молчаливого удаления failures ради ready=true.

## Результаты

Исходная ошибка проверена; полный проход ещё выполняется. Курация не выполнена.


### Проверка реализации курации

Добавлены `MLService/tools/curate_tmj_od3d.py` (143 строки) и
`MLService/tests/test_curate_tmj_od3d.py` (206 строк). Проверка в существующем
Python 3.12 окружении `cloud-pins-venv`, из `MLService` с `PYTHONPATH` корня
репозитория и `MLService`: `python -m pytest tests/test_curate_tmj_od3d.py -q
-p no:cacheprovider` — **25 passed in 0.74s**.

Synthetic fixture использует 20 пациентов вместо 1043 через тестовую замену
константы; это не реальная курация и не оценка качества модели. Проверены
complete/EOF/metadata coverage, совпадение state/index/receipts, точное покрытие
failure reviews, неизвестная причина, missing/mismatched policy, изменённый crop,
поздняя мутация уже скопированного crop или source state, no overwrite/overlap,
лимит размера, приватность исходных идентификаторов и CLI aggregate output.

Утилита принимает только `annotation_side_conflict` с reviewed disposition
`exclude` и непустой приватной причиной. Policy привязана к SHA256 исходного
`preparation.private.json`; provenance сохраняет SHA256 state/index/metadata/policy
и исходные failure counts. `patient_count` считает только пациентов с retained
records; `accepted_source_patient_count` также учитывает prepared patients без
кропов. В новый каталог копируются только известные проверенные NPZ, приватный
индекс и aggregate report; raw processed IDs, receipt rows и free-text reasons
не переносятся. Исходный completed root не изменяется.

Реальные source files и live execution в этой проверке не читались и не менялись.
Реальная курация и binary/multilabel consumer preflight остаются отдельной
приёмкой после завершения полного прохода.

## Закрытие приёмки на выбранной когорте

Пользователь остановил добор и сохранил цель готового запуска; полная загрузка
1043 пациентов больше не обязательна для этого запуска. Уточнённый scope и
фактические gates: [009](../009-bounded-cohort/README.md). План текущего
запуска закрыт на выбранных 148 source cases: два технически противоречивых
пациента исключены с policy/provenance; 146 пациентов и 289 сторон приняты.
Binary/multilabel real preflight и staged Torch 2.6 CPU step прошли; test не
оценивался. Frozen split 102/22/22 пациентов, development-only вне test.
Два immutable пакета DataSphere собраны и проверены установленным CLI.
Full-release EOF не достигнут и не заявляется; source.complete=false сохранён.
Платное обучение и улучшение метрик не заявляются выполненными.
