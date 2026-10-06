# Итоги исследований ВНЧС — 6 октября 2026

Работа приостановлена по просьбе пользователя. Активная цель имеет статус
`paused`; нового обучения сегодня не запускать. Продолжение — только после
указания пользователя. Это сохранённая точка продолжения, не завершение проекта.

## Данные и задача

Публичная TMJ-OD3D V2 с авторской разметкой костных изменений ВНЧС:
146 пациентов /289 сторон. TRAIN102 пациента /204 стороны,
validation22/43, test22/42. Когорта — ограниченный archive prefix,
source complete=false; репрезентативность полного релиза не подтверждена.

Вход текущих моделей — авторские ROI из16срезов на сторону. Binary означает
normal /any osseous pathology; multilabel — шесть независимых признаков,
которые могут сочетаться. Это отдельная задача от положения сустава.
Новые врачебные метки для этих экспериментов не используются.

## Результат на конец дня

Сохраняется frozen ImageNet ResNet18 mean16 + регуляризованный logistic head.

| Проверка | Результат | Решение |
|---|---|---|
| [Nested TRAIN-only binary CV](../osseous_resnet_nested_regularization_20261006/README.md) | Mean patient-fold AUROC0,687931 | Исследовательский кандидат |
| [TRAIN-selected C0,01, original validation](../osseous_resnet_selected_validation_20261006/README.md) | AUROC0,727273; balancedaccuracy0,659091; profile151 AUROC0,571429 | Нет доказанного ranking gain к прежнему C1 |
| [Layer4 fine-tuning на A100](../osseous_layer4_cv_20261006/README.md) | AUROC0,697102; delta+0,009170; regression profile1110,033144 | Заранее заданный критерий не выполнен |
| [Max16 вместо mean16](../osseous_max_features_cv_20261006/README.md) | AUROC0,640263; delta−0,047668; выигрышей2/5 | Max16 не выбран |

A100 worker занял80,05с, provider lifecycle394,693с (~6,6мин).
Оценка только вычислений —59,52RUB; фактический счёт не проверен,
storage/egress не включены. Это стоимость одного job, не всего исследования.
Последний A100 job `bt1ovdhqt335unrmvb7q` повторно проверен через CLI:
provider SUCCESS. CPU max16 завершён exit0; новое multilabel обучение не началось.

Все эти сравнения — development evidence на reused patient folds.
Validation ранее использовалась; исторический test уже был открыт первым scratch
baseline (AUROC0,291). Test не является новым blind holdout, и последующие
layer4/max16 эксперименты его не открывали. AUROC не равен accuracy.

## Что сохранено

Постоянные контракты и [реестр экспериментов](../README.md) ведутся в Git.
Завершённые source/API/result изменения до[PR115](https://github.com/tzopiz/MasterProject/pull/115)
включительно merged; main checkpoint `fd7a83e06f0c30acb8b3bd72869b7d52b0c0fbaa`.
Публичные API frozen features/head и layer4 имеют focused synthetic проверки;
они не являются доказательством клинической пригодности.

Приватно сохранены исходные данные, split/folds, cache, weights, индивидуальные
predictions, manifests, hashes, artifact-verification receipts и budget ledger.
Они не публикуются в GitHub. Отдельный private session-close receipt содержит
точные локальные пути и hashes для возобновления; permissions600/700.

## Точка продолжения

[Шестиметочный протокол](../osseous_multilabel_cv_20261006/README.md) зафиксирован
до fit и независимо проверен; статус planned. Протокол SHA256:
`0e9d617ac87ee3c0f15373514c866c6ea927a8d024b064f3b766af3a04c748aa`.
Prereg commit `15e6b56a84f1bd2b9d5d94f4c8687aae0387c78e`,
ветка `codex/osseous-multilabel-probe`; [change017](../../../docs/changes/017-osseous-multilabel/README.md) остаётся open.

Приватный worker сохранён, но прерван до завершения разработки/ревью:
492строки, SHA256 `a3bec867593d0656b2e8e927b040e6e85bc861cdc2d6e94ecff55826cecdf4e5`.
Синтаксис разбирается; это не подтверждает приёмку или качество реализации.
Fits, OOF predictions и результаты шестиметочного эксперимента отсутствуют.

После возобновления: закончить worker, независимо проверить fixed support masks,
patient isolation, selection/bootstrap и checkpoint replay; только затем запускать
предусмотренный CPU эксперимент. Per-label support и null обязательны для редких
признаков; Other встречается лишь у7TRAIN пациентов. Не заменять binary scores
максимумом шести marginal scores без отдельного протокола.

Дальше остаются public CLI/CV orchestration для воспроизведения после clone и
проверка полного DICOM→localization→classification пути. Пока эти части не доказаны,
готовность «только нажать обучение и получить классификацию любого DICOM» не заявляется.
Автоматического обучения или продолжения на следующий день не назначено.
