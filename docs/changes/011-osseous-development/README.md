# 011 — Разработка osseous моделей без повторного test

Статус: done

## Задача и границы

Первый GPU baseline завершился отрицательным результатом. Пользователь поручил
автономно исследовать улучшение метрик, сохраняя приватность и остановив добор.
Текущий trainer каждый раз открывает test. Требуется отдельный development путь
на том же frozen split; held-out membership не меняется.
Связь: FR-ML-OSSEOUS-DEVELOPMENT, FR-ML-OSSEOUS-EVALUATION.

## Решение и приёмка

- `train(config, development=True)` и CLI `--develop` требуют существующий frozen split.
- Читаются только train/validation NPZ. Test metadata используется только для
  проверки membership/provenance; test payload не проверен этим режимом.
- Epoch/threshold selection остаётся validation-only; checkpoint reload проверяется.
- Отчёт обозначен development, содержит train/validation metrics, без test metrics.
  Private predictions/manifest содержат только development records.
- История содержит train weighted loss, validation weighted loss и probability spread.
- Обычный holdout запуск сохраняет контракт и проверяет все payload.
- Проверки обоих режимов со физически отсутствующими test crops; отказ при отсутствии
  или повреждении frozen split; прежние tests и `scripts/check_specs.py`.

## План

1. Добавить failing behavioral checks с недоступными held-out crops.
2. Реализовать development preparation/evaluation и историю loss.
3. Обновить спецификацию/навигацию; проверить оба training пути.
4. Независимое ревью и отдельный PR. Эксперименты имеют собственный протокол.

## Журнал

2026-10-06: read-only audit не выявил inversion/order/checkpoint ошибки.
Selected checkpoint: train AUROC 0.5333, validation 0.7557; train probabilities
имеют std 0.00339 и близки к 0.5. Validation source-profile composition несбалансирован.
Это диагностические ассоциации, не установленная причина test failure.
Режим реализован. Binary/multilabel при отсутствующих test NPZ завершаются;
CLI --develop проверен. 51 targeted training/cloud tests passed, ruff и specs OK.
Independent review: два замечания (membership до payload; стабильный BCE из logits)
воспроизведены тремя failing tests и исправлены; повторное ревью blockers не выявило.
TRAIN-only диагностический прогон завершён; candidate quality experiment не запускался.
Все четыре шага плана выполнены; PR публикуется отдельно от запуска GPU.
