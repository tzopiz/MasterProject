# 007 — Готовый запуск osseous baseline в DataSphere

Статус: in-progress. Зависимости: [005](../005-od3d-preparation/README.md),
[006](../006-od3d-training/README.md).

## Контракт и план

1. tools/datasphere_osseous.py готовит отдельный immutable payload готовой
   полной когорты, index/crops и frozen patient split. Перепроверяет реальный
   consumer после переноса относительных путей. Не использует position labels.
2. prepare без external IO; plan содержит project/profile/resource/Python,
   runtime/hourlyprice/date/currency/manifest/jobhash/inputhash/splithash.
   Подтверждение точного SHA256 пользуется existing confirm_bundle duplicate
   lock/lifecycle. Подготовка не запускает GPU. Payload≤5GiB.
3. Worker вызывает shared osseous CLI в отдельной группе процессов;
   timeout убивает группу и пишет training-status, не выдаёт successfuljob за
   completemodel. Результаты требуют проверенных completionartifactdigests.
4. Python3.12/pinned researchrequirements, gt4.1, max4hours training.
   Актуальный тариф [официальная таблица](https://yandex.cloud/ru/docs/datasphere/pricing)
   168.48RUB/hour на 2026-10-05; estimate trainingwindow673.92RUB, не hardmoneycap,
   environmentsetup/storage/egress отдельно. Перед платным launch — подтверждение
   конкретного плана. Auth/profile/project проверяются read-only.
5. Tests: staging сохраняет osseous labels и frozen membership,
   mutatedpayloadrefused, неполныйdatasetrefused, zero network prepare,
   timeout/resultintegrity. Готовыйplan хранится приватно; public только агрегаты.

## Результаты

Read-only datasphere project get вернул id/name/community_id для существующего
проекта профиля tmj-master. GPU/job пока не запускались.

Пять focused cloud tests пройдены на Torch 2.6.0 CPU. Ревью обнаружило
недостаточную привязку возвращаемых результатов к ожидаемому запуску;
исправлено: completion/report проверяются против bindings staged плана,
включая input, split, task, codebook и target mapping.

Cloud requirements закреплены на Torch 2.6.0+cu118: это соответствует
[документированному Jobs image](https://yandex.cloud/en/docs/datasphere/concepts/jobs/docker)
CUDA 11.8 и [официальной матрице PyTorch](https://pytorch.org/get-started/previous-versions/).
Проверка CPU той же версии не выдаётся за проверку GPU/driver.


### Полный индекс и явный lifecycle CLI

Независимый review воспроизвёл блокер: валидный синтетический индекс
1043 пациентов / 2086 side records (1097399 bytes) проходил research preflight,
но cloud preparation отказывала из-за generic `_read` cap 1 MiB. Исправлено
только в cloud source-index boundary: `_read_index` ограничен 16 MiB и вызывается
до consumer preflight; lifecycle/config `_read` с лимитом 1 MiB не изменён.
Reader отказывает при duplicate JSON keys, invalid UTF-8, NaN/Infinity и
overflow literals вроде 1e999; безопасный код `invalid_private_index`.

CLI теперь предоставляет status, cancel, results и reconcile через существующие
lifecycle функции. Для reconcile необходимы job-id, operation-id и явный
--acknowledge-match; команда не повторяет submission. Job environment и план
явно фиксируют docker_image `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`,
согласованный с Torch 2.6.0+cu118; скрытая зависимость от platform default устранена.

Проверка: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MLService/docs/ai/tasks/GH-85/cloud-pins-venv/bin/python -m pytest -p no:cacheprovider
MLService/tests/test_datasphere_osseous.py -q --tb=short` из корня —
**20 passed за 3.13 с**. Полный синтетический индекс staged без provider IO,
точное equality index и frozen split проверены. TDD red наблюдался для старого
cap, непредоставленных CLI actions, отсутствующего docker pin и численного overflow.
Ruff F/I checks двух изменённых файлов, `python3 scripts/check_specs.py` и
`git diff --check` — PASS. Generic cloud module и image preparation не изменены;
живой процесс подготовки не затронут, downloads/GPU/jobs не запускались.
Это локальная проверка, не live CUDA/CLI transport acceptance.

Дополнительная проверка настоящим установленным DataSphere CLI:
`parse_config(job.yaml)` и `validate_paths(config)` на синтетическом bundle
прошли; Python environment fully-manual, версия 3.12, resource gt4.1,
явный Docker image CUDA 11.8. Provider calls при этой проверке отсутствовали.
Официальные API restrictions проекта/сообщества подтвердили ALLOW_JOBS и
ALLOW_SPEC_GT_4_1; ответы сохранены приватно.
