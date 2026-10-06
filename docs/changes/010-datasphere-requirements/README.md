# Совместимость requirements с DataSphere CLI

Статус: done

Спецификации: [FR-ML-OSSEOUS-CLOUD](../../../MLService/docs/spec/functional/osseous/README.md#fr-ml-osseous-cloud).

## Задача

При разрешённом пользователем бинарном запуске 2026-10-06 официальный CLI отверг комментарии requirements.txt до вызова create job. В проекте найден только старый запуск 2026-09-27. Исходная попытка и lock остаются приватными; повторного confirm этого bundle нет.

## Границы

Удалить комментарии из cloud requirements, перенести пояснение в документацию. Официальный CLI 0.10.0 также превращает manual local-paths=[] в None и падает в validate_inputs до create. Указать один уже объявленный launcher payload/tools/run_osseous_research.py как local module; дополнительных данных нет, повторно упакованный файл учесть в estimate. Версии пакетов, GPU, обучение, данные, split и стоимость не менять. Новый immutable bundle с новым digest; старый сохранить.

## Требования

Конфигурация должна проходить настоящий offline define_py_env, validate_paths, prepare_local_modules и check_limits установленного официального CLI; package specifiers должны оставаться torch 2.6.0+cu118, numpy 2.1.3, scipy 1.14.1, scikit-learn 1.6.1 с прежним CUDA index. Данные, split, training code и версии остаются прежними. Ошибка до create не разрешает удаление submission lock.

## План

- [x] Воспроизвести отказ и добавить регрессионную проверку requirements.
- [x] Удалить комментарии, обновить baseline и реестр.
- [x] Проверить tests, спецификации, официальный offline parser и новый private bundle; сохранить frozen bindings.

## Проверка

До исправления: официальный CLI packaging parser завершил InvalidRequirement на первой строке комментария; execution receipt пустой. Job list содержит только старый SUCCESS 2026-09-27. После исправления: целевой regression test падает до исправления и проходит после; `python -m pytest MLService/tests/test_datasphere_osseous.py -q`: 21 passed. Установленный официальный CLI 0.10.0: parse_config → define_py_env → validate_paths → prepare_inputs → prepare_local_modules → check_limits → get_job_params прошли offline без API. Новый приватный bundle сохраняет все training/data/split bindings; изменились только requirements, cloud launcher и job config. Обе неудачные попытки сохранены с lock; повторный read-only list снова показывает только старый запуск.

## Результат

Совместимость локальной упаковки исправлена; training, dataset, split, версии и цена прежние. Платный запуск отдельного нового bundle разрешён пользователем; этот документ не объявляет облачное обучение успешным.
