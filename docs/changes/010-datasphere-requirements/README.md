# Совместимость упаковки и среды DataSphere

Статус: done

Спецификации: [FR-ML-OSSEOUS-CLOUD](../../../MLService/docs/spec/functional/osseous/README.md#fr-ml-osseous-cloud).

## Задача

Разрешённый пользователем бинарный запуск 2026-10-06 выявил ошибки среды: CLI 0.10.0 отвергал комментарии requirements и терял пустой manual local-paths; внешний CUDA image не содержал Python; platform pip в системном контейнере смешивал pinned NumPy со старым pandas и завершал import sklearn ошибкой ABI. Обучение в ошибочных job не начиналось.

## Границы

Источник проблемы — сборка окружения, а не reader модели. Финальный путь: plain system-python-3-10 image и bootstrap внутри immutable payload. Не объявлять env.python или дополнительные local modules. Сохранить старые bundles/locks/terminal receipts; каждый исправленный запуск использует новый bundle после установления исхода прежнего. Данные, frozen patient split, training code, GPU, ML-пины и ставка прежние. Python baseline явно меняется с requested 3.12 на наблюдаемый 3.10.12; biological/anatomical validation в объём не входит.

## Требования

Bootstrap очищает PYTHONPATH/PYTHONHOME, проверяет Python 3.10, создаёт venv без system-site-packages/ensurepip и устанавливает пакеты через pip --python. Перед worker обязательны успешные imports и CUDA; фактические версии фиксируются в приватном platform log. Пины: Torch 2.6.0+cu118, NumPy 2.1.3, SciPy 1.14.1, scikit-learn 1.6.1. Полный официальный offline CLI путь должен пройти без platform Python setup. Provider SUCCESS принимается только вместе с complete training, exit 0, совпадающими frozen bindings и artifact digests. Качество модели оценивается отдельно и не подменяет исправление bootstrap.

## План

- [x] Воспроизвести отказы и проверить источник записи окружения.
- [x] Реализовать isolated bootstrap, обновить спецификацию и реестр.
- [x] Проверить локальную изоляцию, CLI упаковку, реальный CUDA запуск и приватные результаты.

## Проверка

Requirements regression падает до исправления; после: 21 targeted cloud test passed. Bash syntax, specs, lint/format тестов и git diff check проходят. Реальный local smoke с загрязняющим pandas в PYTHONPATH: venv --without-pip и pip --python работают, pandas/NumPy системного окружения не наследуются.

Официальный установленный CLI 0.10.0: parse_config → validate_paths → prepare_inputs → check_limits → get_job_params проходят offline с python_env отсутствующим. Prepared bundles сохраняют все input/split/training bindings. Выполненные source files совпадают с итоговым PR побайтно; никаких данных вне payload не объявлено.

Реальный isolated job подтвердил Python 3.10.12, Torch 2.6.0+cu118, NumPy 2.1.3, SciPy 1.14.1, scikit-learn 1.6.1, CUDA 11.8, cuda_available=true и использование GPU. Provider SUCCESS, worker complete/exit 0, results_ready=true; completion/report/checkpoint/predictions hashes и frozen bindings проверены после приватного download.

## Результат

Приёмка технического запуска выполнена. Первый binary run дал отрицательный результат качества на oracle-ROI: ниже constant baseline; агрегаты и ограничения в [протоколе](../../../MLService/experiments/tmj_od3d_feasibility_20261005/README.md#результат-первого-binary-training-2026-10-06). Веса, predictions, logs и receipts остаются приватными. Исправленный multilabel пакет подготовлен локально, но не отправлялся; новый платный эксперимент не запускался.
