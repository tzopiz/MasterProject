# 012 — Экспериментальная архитектура osseous

Статус: done

## Проблема и целевое изменение

TRAIN-only capacity probe на одних 16 пациентах / 32 сторонах, CPU Torch 2.11,
1600 updates дал spatial AUROC 1 / BCE 0.1374 против global 0.703 / 0.6739.
Это мотивация доступности варианта для воспроизведения. Последующий patient CV
не выполнил критерий выбора; spatial не выбран для validation.
Текущий trainer всегда строит global spatial pooling и не хранит имя архитектуры.

Добавить явный `architecture`: `global_mean` (точный прежний default) или
экспериментальный `spatial_head`. Вариант наследует SliceBagClassifier, сохраняет
conv trunk, меняет только последнюю pooling на 4×4 и head на
Linear(256,32), ReLU, Linear(32,outputs); mean pooling срезов наследуется.
Binary даёт один output, multilabel — шесть. Параметры обучения, ограничения,
данные, seed, зависимости, аугментация и приватность не меняются.
Связь: FR-ML-OSSEOUS-ARCHITECTURE, FR-ML-OSSEOUS-EVALUATION.

## Приёмка

- Requested architecture используется при обучении и восстановлении checkpoint;
  metadata/config сохраняют имя, неизвестные или несовпадающие имена дают
  фиксированный безопасный ResearchError.
- Старый checkpoint без имени восстанавливает global_mean; global bindings
  сохраняют совместимость immutable bundles. Spatial binding фиксирует имя.
- Binary/multilabel development работают без test crops; probabilities после
  reload равны сохранённым; mean bag сохраняет инвариантность перестановки.
- Cloud consumer отклоняет неизвестную/несогласованную архитектуру и принимает
  старые артефакты без имени. Никакой оценки качества по synthetic checks.

## План

1. Добавить синтетические failing behavioral tests.
2. Реализовать выбор архитектуры и provenance/reload/cloud validation.
3. Обновить постоянные правила, навигацию и реестр.
4. Выполнить focused CPU checks, ruff, specs и diff validation; передать ревью.

## Результаты

2026-10-06: все четыре шага выполнены. Независимое code/acceptance review
относительно ceacc86 не выявило блокеров; reviewer повторил focused suite:67passed
за7,43s (единственное предупреждение — недоступен pytest cache). Публикация не выполнялась; вариант сохраняется для воспроизведения отрицательного CV.
TRAIN-only grouped CV не прошёл предзаданный go/no-go: spatial не выбран для
следующей validation и остаётся только явным experimental вариантом.

Red: новый файл test_osseous_architecture.py дал 15 failed на исходном коде;
после реализации trainer — 13 passed / 2 failed из-за cloud consumer,
принимавшего неизвестное имя с согласованным digest. Consumer исправлен.
Green: `MLService/docs/ai/tasks/GH-85/cloud-pins-venv/bin/python -m pytest -q
MLService/tests/test_osseous_architecture.py MLService/tests/test_tmj_osseous_research.py
MLService/tests/test_datasphere_osseous.py` — 67 passed (финальный запуск 6.37s).
Только синтетические bags; binary/multilabel development без test NPZ, replay,
legacy/default exact logits, slice order, конфликты и cloud staging проверены.

Ruff check tests/trainer и format обоих изменённых test файлов прошли.
В datasphere_osseous.py осталось ровно прежних 58 lint findings
(E701=25, E702=32, E721=1); добавленный код новых findings не создаёт.
`git diff --check` и `python3 scripts/check_specs.py --self-test` прошли.
Обычный specs checker отклоняет ссылку на ignored/untracked change012;
при временном Git index, содержащем только этот дополнительный README,
`python3 scripts/check_specs.py` прошёл. Настоящий index не изменён.
Private data, реальные training jobs, downloads и cloud не использовались.


Итоговое ревью: legacy/default exact numerics и global bindings, spatial binary/
multilabel development без test payload, checkpoint/conflict и cloud consumer
проверены. Публикация не подтверждает качество spatial; отрицательный CV сохранён.

После точного добавления change012 в настоящий Git index: `python3 scripts/
check_specs.py`, `--self-test` и cached diff check — PASS. Ruff tests/trainer и
format tests выполнены с --no-cache; permission на запись cache не нужен.
