# Совместимость упаковки и среды DataSphere

Статус: done

Спецификации: [FR-ML-OSSEOUS-CLOUD](../../../MLService/docs/spec/functional/osseous/README.md#fr-ml-osseous-cloud).

## Задача

При разрешённом пользователем бинарном запуске 2026-10-06 официальный CLI отверг комментарии requirements.txt до вызова create job. В проекте найден только старый запуск 2026-09-27. Исходная попытка и lock остаются приватными; повторного confirm этого bundle нет.

Облачный job завершился ERROR до training: внешний nvidia/cuda image не содержит Python. Использовать документированный системный образ system-python-3-10 с Conda; реальные логи показывают Python 3.10.12: provider предупреждает и использует системный Python вместо запрошенного 3.12. Следующие планы явно указывают 3.10, совпадающий с контейнером и проверенный текущей CI matrix. Текущий immutable job остаётся неизменным; фактический runtime записан в приватный audit и журнал. Ссылка на [Docker images](https://yandex.cloud/en/docs/datasphere/concepts/jobs/docker).

Системный job затем завершил import sklearn ошибкой pandas/NumPy ABI: automatic requirements установлены в global site-packages поверх старых библиотек образа. Целевой writer среды: plain system container без env.python, bootstrap создаёт venv --without-pip и направляет pip --python в неё. PYTHONPATH/PYTHONHOME очищаются; данные и ML-пины прежние. Требование: фактический isolated interpreter импортирует ML-пакеты и видит CUDA до worker.

## Границы

Удалить комментарии из cloud requirements, перенести пояснение в документацию. Официальный CLI 0.10.0 также превращает manual local-paths=[] в None и падает в validate_inputs до create. Указать один уже объявленный launcher payload/tools/run_osseous_research.py как local module; дополнительных данных нет, повторно упакованный файл учесть в estimate. Версии ML-пакетов, GPU, обучение, данные, split и стоимость не менять. Новый immutable bundle с новым digest; старый сохранить.

## Требования

Конфигурация должна проходить настоящий offline parse_config, validate_paths, prepare_inputs, check_limits и get_job_params официального CLI без env.python; установка ML-пакетов выполняется bootstrap в isolated venv; package specifiers должны оставаться torch 2.6.0+cu118, numpy 2.1.3, scipy 1.14.1, scikit-learn 1.6.1 с прежним CUDA index. Данные, split, training code и версии ML-пакетов остаются прежними. Изменение Python 3.12 → 3.10 явно записывается; текущий runtime подтверждается platform log, не ожидаемым значением job config. Ошибка до create не разрешает удаление submission lock.

## План

- [x] Воспроизвести отказ и добавить регрессионную проверку requirements.
- [x] Удалить комментарии, обновить baseline и реестр.
- [x] Проверить исправленный системный контейнер в реальном запуске после terminal ERROR прежнего; проверить tests, спецификации, официальный offline parser и новый private bundle; сохранить frozen bindings.

## Проверка

До исправления: официальный CLI packaging parser завершил InvalidRequirement на первой строке комментария; execution receipt пустой. Job list содержит только старый SUCCESS 2026-09-27. После исправления: целевой regression test падает до исправления и проходит после; `python -m pytest MLService/tests/test_datasphere_osseous.py -q`: 21 passed. Установленный официальный CLI 0.10.0: parse_config → define_py_env → validate_paths → prepare_inputs → prepare_local_modules → check_limits → get_job_params прошли offline без API. Новый приватный bundle сохраняет все training/data/split bindings; изменились только requirements, cloud launcher и job config. После облачной ошибки меняется только выбор контейнера, без смены GPU/пинов/протокола. Обе неудачные попытки сохранены с lock; повторный read-only list снова показывает только старый запуск.

## Результат

Историческая manual установка дошла до imports и завершилась pandas/NumPy ABI error. Финальный plain bootstrap проверен локально: venv --without-pip плюс pip --python работает, synthetic PYTHONPATH с загрязняющим pandas не наследуется. Bash syntax и 21 cloud test прошли. Реальный isolated job подтвердил Python 3.10.12, Torch 2.6.0+cu118, NumPy 2.1.3, SciPy 1.14.1, scikit-learn 1.6.1, CUDA 11.8 и cuda_available=true без Traceback. Runtime сведён с plan, приватные receipts сохранены. Training completion и качество не являются приёмкой bootstrap и пока не объявляются. Совместимость локальной упаковки исправлена; training, dataset, split, версии ML-пакетов и цена прежние; Python baseline явно согласован с наблюдаемым системным runtime 3.10. Платный запуск отдельного нового bundle разрешён пользователем; этот документ не объявляет облачное обучение успешным.
