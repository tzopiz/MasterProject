# MLService / tests

Юнит- и интеграционные тесты на **pytest**. Все исследовательские сценарии ниже используют синтетические данные, временные каталоги и CPU.

## Полный набор

Из `MLService/` с активированным окружением общих зависимостей:

```bash
python -m pytest tests/ -v --tb=short
```

CI сохраняет проверку существующего приложения на Python 3.10/3.11/3.12. Отдельный исследовательский job использует Python 3.12 и узкий набор зависимостей, чтобы зависимости приложения не скрывали отсутствующие импорты в DataSphere payload.

## Воспроизводимый research smoke

Подготовить отдельное Python 3.12 окружение. На Linux CPU сначала установить `torch==2.11.0` из официального CPU index, затем `tools/research_cloud_requirements.txt` и `pytest==9.1.1`. Это соответствует [официальной команде PyTorch 2.11 для CPU](https://pytorch.org/get-started/previous-versions/). На macOS использовать CPU-совместимый пакет той же версии. CUDA и MPS для этого smoke не нужны.

Из `MLService/` одна команда повторяет исследовательский job CI:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -p no:cacheprovider \
  tests/test_canonical_training_inputs.py \
  tests/test_roi_provenance.py \
  tests/test_run_research.py \
  tests/test_sagittal_binary_cv.py \
  tests/test_cv_artifacts.py \
  tests/test_multiclass_research.py \
  tests/test_infer_research.py \
  tests/test_research_architecture.py \
  tests/test_datasphere_research.py \
  -q --tb=short
```

Команда использует существующие проверки, без второго trainer или отдельного orchestration layer:

| Проверки | Что доказывают |
|---|---|
| Canonical inputs, ROI provenance, runner | Строгий intake, source-qualified patient grouping, пригодность folds до создания модели, DICOM geometry/order, фиксированный crop/padding, паспорта/cache, ограниченный профиль архитектуры до allocation и безопасные отказы |
| Binary CV artifacts и multiclass research | Реальные CPU optimizer steps меняют веса; checkpoints восстанавливают предсказания; сохранённые private rows воспроизводят метрики; выбор модели остаётся development CV |
| Research inference | Реальная DICOM-серия и парные heatmap weights дают две ROI и две sagittal оценки через все сохранённые folds, в обоих режимах; исходные training index/crops предварительно удалены |
| DataSphere research | Offline prepare/rekeying, контроль подтверждения, duplicate/ambiguous submission, timeout/results guards; staged worker действительно обучает на CPU и сохраняет артефакты |

Транспорт DataSphere и Drive в тестах заменён управляемыми doubles: команда не запускает платные jobs, не скачивает медицинские данные и не проверяет доступность облачной инфраструктуры. Worker, ROI writer, train/save/reload и вычисление метрик выполняются реально. CLI проверяется на некорректных аргументах и synthetic sensitive tokens.

Все patient-level rows, linkage/replay и inference results остаются private; даже псевдонимизированные данные не считаются анонимными. Тестовые артефакты pytest находятся во временных каталогах. CI не публикует их.

## Проверенное выполнение

2026-10-03, macOS arm64 CPU, Python 3.12.14: команда smoke выше прошла **286 тестов за 25.38 с** в изолированном окружении pinned research dependencies (PyTorch 2.11.0). Полный набор отдельно прошёл **414 тестов за 35.15 с** в основном окружении (PyTorch 2.14.1). Ruff 0.16.10 check/format, проверка спецификаций и `git diff --check` прошли. Это локальное подтверждение; запуск нового Linux CI job и CUDA эксперимент этой проверкой не выполнялись.

## Границы результата

Успешный smoke подтверждает техническую связность pipeline и безопасные отказы. Он не устанавливает качество обученной модели, независимость обучения детектора или итоговую метрику на реальной когорте. Локальная проверка macOS CPU не подтверждает выполнение на Linux/CUDA; проверка Linux CPU предусмотрена в job CI, CUDA требует отдельного реального запуска. Seed не гарантирует побитовую идентичность между платформами и версиями библиотек.

Ruff `0.16.10` закреплён одинаково в `requirements-dev.txt` и pre-commit. CI и hooks проверяют существующие app/test directories и явный список исследовательских модулей; неподготовленные legacy tools не включаются автоматически.
