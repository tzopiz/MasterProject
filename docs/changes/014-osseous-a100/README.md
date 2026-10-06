# 014 — A100 для ускоренных исследовательских запусков

Статус: done

## Проблема и целевое изменение

Cloud package ограничен gt4.1/T4. Для более тяжёлых гипотез пользователь
поручил использовать мощнее GPU в согласованном приватном бюджете и сроке.
Read-only DataSphere API2026-10-06 подтвердил ALLOW_SPEC_G_2_1/ALLOW_JOBS;
это разрешённая конфигурация, не гарантия свободной capacity.

Добавить только g2.1/A10080GB наряду с прежним gt4.1. Published rates inclVAT
на2026-10-06:168,48/542,88RUBperhour respectively
([официальный тариф](https://yandex.cloud/ru/docs/datasphere/pricing)).
Optional quote по умолчанию выбирается по resource; explicit positivefinite
override допустим. Цена T4 не должна молча применяться к A100.
Training/bootstrapping/confirmation и artifact semantics не меняются.
Связь FR-ML-OSSEOUS-CLOUD.

## Приёмка

- API/CLI принимают ровно gt4.1 иg2.1; defaultT4 сохранился.
- Job и immutableplan используют одну выбранную конфигурацию и rate.
- Training-windowestimate корректен; setup/storage/egress по-прежнему отдельно,
  hardmoneycap не заявляется. Приватная бюджетная reservation ведётся отдельно.
- Invalid resource/rate отклоняется safeCloudLaunchError до создания bundle;
  staging не вызывает provider IO.
- Synthetic staging/CLI/quote/default и digest checks проходят, independentreview.
  Приёмка кода не означает уже выполненный A100 запуск.

## План

1. Failing synthetic checks для новогоresource и rate selection.
2. Реализовать узкий API/CLI selection и safequote validation.
3. Обновить cloudspec/nav и выполнить focused checks/review отдельным PR.

## Результаты

2026-10-06: все три пункта выполнены; source+18/-3,tests+133/-0.
Default price/date resource-specific: прежний T4Oct5,новый A100Oct6;
explicit timezone-aware date сохранён. Independent review закрыто:
sourceSHA a23663f69cfe7f52f73320ee8e8a15a41a80348c38d6ec3ecc46ca24a7812f36,
testSHA d22866c3a1e57c293399a12f67c0af544693d8579c8033ab2890e83f6b31676c.

Фактически: focusedtestfile42passed (peer4,48s); testRuffpassed,
source58preexistingfindings не изменились; diffcheckpassed. Datefix red4failures
перед green. Provider IO/privatecohort/paidjobs в synthetic проверках не было.
Cloudspec/nav обновлены. Приёмка кода не подтверждает A100 запуск/скорость.
Пользовательский бюджет/грант остаются в приватном плане.
