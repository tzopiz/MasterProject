# Подготовка и запуск исследования в DataSphere Jobs

Контракт — [FR/TC-ML-READINESS-CLOUD-LAUNCH](../spec/features/dataset-readiness/README.md#fr-ml-readiness-cloud-launch). Реализация — [datasphere_research.py](../../tools/datasphere_research.py); проверки — [синтетические сценарии](../../tests/test_datasphere_research.py). Launcher использует официальный CLI; подготовка не обращается к облаку и не запускает обучение.

## Вход и приватный план

Нужны приватный schema-v1 config [единого research runner](../../google_colab/README.md), готовый canonical index и пары ROI с валидными паспортами. Локальная папка данных подходит без повторного скачивания. Для предоставленной Google Drive **папки** есть отдельная команда; она переносит данные локально, но не создаёт patient mapping, метки или геометрию. Ссылка на отдельный файл, архив другого провайдера и недостаточные права требуют заранее подготовленного локального корня. URL хранится в локальном приватном `.txt`, не в Git или команде shell:

```bash
cd MLService
python tools/datasphere_research.py download-source \
  --url-file data/source-link.private.txt --destination data/staged-source
```

Команда использует установленный `gdown --folder --output`; `--gdown` позволяет указать его executable. Это явно запрошенный перенос, не часть offline prepare. Новый destination обязателен; при ошибке частичные файлы остаются приватными для проверки, автоматического retry/слияния нет. Доступ, квоты и network проверяются по фактическому источнику; снимки не скачивались для проверки launcher. Официальное описание: [Google Drive](https://yandex.cloud/en/docs/datasphere/operations/data/connect-to-google-drive).

`prepare` выполняет настоящий preflight runner, включая группировку пациентов, классы в CV folds и ROI-паспорта. В новую папку bundle он копирует только пары crop/passport, необходимый Python-код из фиксированного списка, зависимости и новый технический index/config. У index новые случайные plan-local source/study/patient/label tokens; повторные исследования одного source-qualified patient остаются одной группой. Series paths удалены. Study key и input digest паспортов пересчитаны для новых tokens; источник, физическая/voxel геометрия и pixel bytes сохраняются. Новый prepare создаёт новые tokens и меняет сортировку canonical records/groups: тот же seed может дать другое fold membership относительно исходного local index или другого bundle. Для точного повторения и CPU/cloud сравнения используют один frozen payload/index/config; приватные membership/digests сохраняются в артефактах. Исходное сопоставление хранится только в `mapping.private.json` **вне** объявленного upload input.

В облако объявлен только каталог `payload`. `plan.private.json`, original mapping, project ID, исходный config/run ID, task runtime, Git, документы и DICOM не входят в inputs. Remote run ID — `research`, пути config относительные, outputs отдельно от inputs. Удалённый `device=null`/unset разрешается в `cuda`: недоступная CUDA вызывает отказ, без молчаливого перехода на CPU. Для намеренного CPU-запуска укажите `device="cpu"`; macOS-only `mps` отвергается до staging. NIfTI проверяется общим ROI validator: text/extension fields, gzip filename/comment/extra metadata, timestamp, дополнительные members и хвостовые данные запрещены; распаковка ограничена объявленным объёмом float32. Это минимизация и псевдонимизация; пиксели, лица, метки и групповые prediction rows остаются приватными. Bundle, скачанные результаты и логи хранить в игнорируемом `MLService/data/`; ничего из них не публиковать.

## Один явный confirm

Локальный launcher требует Python 3.12 с [исследовательскими зависимостями](../../tools/research_cloud_requirements.txt). DataSphere CLI должен быть установлен и авторизован через `yc` по [официальной инструкции](https://yandex.cloud/en/docs/datasphere/operations/projects/work-with-jobs). Настройка аккаунта и права проекта не меняются launcher. Если проверенный аккаунт использует named profile, передать его в prepare как `--profile "$DS_PROFILE"`: имя сохранится только в приватном плане и будет использовано как global `--profile` официального CLI во всех lifecycle-командах. Без него действует native default, который оператор должен проверить. Эти значения задаёт оператор приватно: `DS_PROJECT_ID`, ставка `DS_HOURLY_PRICE`, трёхбуквенная валюта `DS_CURRENCY`, timezone-aware время актуальности `DS_PRICE_AS_OF`. Ставка не подставляется из чужого аккаунта.

```bash
python tools/datasphere_research.py prepare \
  --config data/research.private.json --bundle data/cloud-bundle \
  --project-id "$DS_PROJECT_ID" --profile "$DS_PROFILE" --resource gt4.1 \
  --max-runtime-seconds 7200 --hourly-price "$DS_HOURLY_PRICE" \
  --currency "$DS_CURRENCY" --price-as-of "$DS_PRICE_AS_OF"
```

Вывод — digest, количество studies/patients, байты, ресурс/Python, training runtime, предоставленная ставка/валюта/время, training-window compute estimate и исключённые затраты. Полный manifest и проект доступны только в приватном плане. Оператор проверяет эти условия, transfer в нужный приватный проект и записывает **выведенный** digest в `DS_PLAN_SHA256`. Затем:

```bash
python tools/datasphere_research.py confirm \
  --bundle data/cloud-bundle --plan-sha256 "$DS_PLAN_SHA256"
```

`--cli` у confirm/status/results/cancel/reconcile задаёт путь официального executable. Confirm повторно проверяет digest плана, job config и каждого payload-файла; изменённый/добавленный файл или symlink запрещён. Payload имеет read-only permissions. До внешнего вызова `O_EXCL` создаёт `submission.private.json`; параллельный/повторный confirm не создаст второй job. Официальный вызов — `project job execute ... --async -o execution.private.json`; stdout/stderr CLI не печатаются, execution JSON читается из файла. Максимальный payload — 5 GiB до сжатия (консервативный per-input лимит). Превышение требует отдельного согласованного дизайна разделения, не скрытого запуска нескольких jobs.

Если вызов прервался, вернул ошибку или не дал валидный receipt, состояние `ambiguous`: job уже мог быть создан. Lock сохраняется, автоматического retry нет. Сначала проверить jobs официальным CLI в исходном проекте, сопоставить время и фактический job/config с планом; **не** удалять lock ради повторной отправки. После ручного подтверждения соответствия существующего job:

```bash
python tools/datasphere_research.py reconcile --bundle data/cloud-bundle \
  --job-id "$DS_JOB_ID" --operation-id "$DS_OPERATION_ID" --acknowledge-match
```

Reconcile только читает существующий job и привязывает его к receipt; он не способен сам доказать соответствие digest по урезанному ответу `job get`. Если job нельзя однозначно установить, оставить ambiguous и разрешить неопределённость через платформу/оператора.

## Наблюдение, результаты и остановка

```bash
python tools/datasphere_research.py status --bundle data/cloud-bundle
python tools/datasphere_research.py results --bundle data/cloud-bundle \
  --destination data/cloud-results
python tools/datasphere_research.py cancel --bundle data/cloud-bundle
```

Results вызывает официальный `download-files --with-logs` в новый приватный destination; существующая папка не перезаписывается. При сетевой ошибке partial output остаётся для проверки; повторная попытка возможна только в новую папку. Cancel означает запрос отмены, не доказательство мгновенной остановки биллинга.

Cloud status и training status различаются. Wrapper ограничивает время **подпроцесса обучения**, при timeout убивает его process group и сохраняет `results/training-status.json`: `complete` с exit code 0, `failed` или `timeout` с фактическим кодом. Wrapper возвращает управление платформе ради сохранения partial outputs; cloud `SUCCESS` не доказывает complete training. Results выводит training status/code и `results_ready`: true только при complete training, полном отчёте и наличии replay/всех fold checkpoints/private prediction files с совпадающими хешами. Завершённое обучение с частично скачанными артефактами даёт `results_ready: false`; это не пригодный полный ensemble. Официальный CLI фильтрует суммарный download выше 1 GiB и может вернуть 0 при ошибке отдельных файлов; отсутствие файлов не маскируется успехом. Без status-файла возвращается `no_result`; backend failure может не дать никаких артефактов. `training.private.log`, replay config, per-patient prediction rows и checkpoints остаются приватными. Перед чтением отчёта проверить `complete` и полный набор CV fold checkpoints; не трактовать partial run как полноценный ensemble.

## Ресурсы, стоимость и границы проверки

Jobs запускаются на отдельной Linux x86_64 VM, не в notebook kernel; manual environment задаёт Python 3.12, точные direct dependency versions и `local-paths: []` ([официальный runtime](https://yandex.cloud/en/docs/datasphere/concepts/jobs/environment)). Пины — кандидат Linux Jobs среды с torch 2.11.0 из [официального релиза](https://dev-discuss.pytorch.org/t/pytorch-2-11-0-general-availability/3328), не копия macOS task venv и не транзитивный wheel/hash lock. CUDA, доступность GPU/сборка окружения и фактическая ставка проверяются на целевом проекте перед реальным confirm. Результат runner фиксирует фактические версии.

`gt4.1` — предлагаемый один GPU ресурс, оператор может явно выбрать другой. Watchdog не ограничивает время установки/transfer, backend termination, storage, egress или деньги. Estimate равен предоставленной ставке × training window; setup compute/cache/logs/results/egress исключены и явно перечислены. [Бюджетные уведомления](https://yandex.cloud/en/docs/datasphere/concepts/budget) не выключают ресурсы; пустые project limits не доказывают денежный cap. Перед confirm актуализировать [цены](https://yandex.cloud/en/docs/datasphere/pricing), retention и квоты. [CLI/Jobs](https://yandex.cloud/en/docs/datasphere/concepts/jobs/cli) описывает default TTL 14 дней; нужен своевременный приватный download/cleanup. Launcher не создаёт dataset resources и не предполагает notebook mount.

Синтетические проверки покрывают локальное staging/preflight/rekey, immutable digest, повтор/ambiguity, safe CLI/response boundaries, реальные tiny CPU train/status и timeout/failure subprocess. Формат single-entity execution/get JSON (dict с job_id/operation_id или id/status) и распаковка declared `results` ZIP в `results/training-status.json` дополнительно проверены локальными formatter/archive функциями установленного официального CLI, без сетевых запросов. Пины отдельно прошли синтетические проверки на macOS CPU; это не Linux CUDA проверка. Cloud transport заменён doubles; Jobs, upload, реальная Google Drive загрузка и Linux CUDA не запускались. Проверенный ранее read-only доступ к проекту не подтверждает право/ресурсы платного запуска. Геометрический паспорт не устанавливает anatomical ROI quality или независимость от detector-training пациентов.

## Отдельный запуск костных изменений

Для авторских osseous ROI используется `tools/datasphere_osseous.py`; position
launcher выше не принимает эти коды. Контракт и границы: [osseous](../spec/functional/osseous/README.md).
Сначала завершить `tools/prepare_tmj_od3d.py` и consumer preflight; partial
индекс с `complete=false` или необработанными failures не допускается к staging.
Конфигурация приватная, пути разрешаются относительно её файла.

Из каталога MLService:

```bash
python -m tools.run_osseous_research --config /private/path/research.private.json --preflight
python tools/datasphere_osseous.py prepare --config /private/path/research.private.json --bundle /private/path/bundle --project-id PROJECT_ID
```

Prepare не обращается к облаку. Перед confirm оператор проверяет SHA256
плана, task/mode, partitions, байты upload, resource, runtime и цену на дату.
Confirm отправляет ровно эти bytes; никаких raw DICOM или исходных CSV с
демографией в payload нет. Изменение payload требует нового bundle.

Текущая реализация/проверки: [007](../../../docs/changes/007-osseous-cloud/README.md).
Два реальных bundle выбранной пользователем когорты готовы: 148 source cases,
146 retained patients / 289 sides. Оба real preflight, frozen split, staged CPU
step и официальный CLI config validation пройдены; scope и evidence в
[009](../../../docs/changes/009-bounded-cohort/README.md). Source.complete=false
сохранён; полный релиз не выдаётся за полученный. Для остановленного checkpoint
нужна явная bounded policy `cohort={kind:archive-prefix, source_patient_count,
source_end_offset}` в `curate_tmj_od3d.py`, связанная с SHA256 source state и
каждым reviewed failure. Публичных patient rows в документах нет.
Обучение и upload не запускались; стоимость не списывалась за наш запуск.

Официальный CLI разбирает каждую непустую строку requirements-file как package
specifier либо поддерживаемый pip flag; комментарии `#` здесь запрещены.
Osseous requirements используют Python 3.12 и Torch 2.6.0+cu118 с CUDA 11.8,
совместимой с выбранным контейнером. Пояснения находятся в README, а не в
requirements-file. Регрессия и offline parser проверены в [010](../../../docs/changes/010-datasphere-requirements/README.md).

CLI 0.10.0 не сохраняет пустой manual `local-paths`: osseous job объявляет
один launcher `payload/tools/run_osseous_research.py` из проверенного manifest.
Его повторная упаковка как local module учтена в upload estimate; новые
данные или каталоги вне payload не добавляются.
