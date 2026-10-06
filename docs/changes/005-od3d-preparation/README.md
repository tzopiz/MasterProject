# 005 — Подготовка изображений TMJ-OD3D

Статус: in-progress. Основание: [004](../004-od3d-intake/README.md).

## Контракт

Подготовить воспроизводимые приватные numpy NPZ bags из авторских ROI:
16 доступных срезов, 96x96, float16. Это oracle ROI исследование костных
изменений, не end-to-end локализация и не положение сустава.
Выбирать равномерные квантили списка доступных пар box/InstanceNumber;
при малом количестве повторять доступные срезы. Не интерполировать пропуски
серии и не трактовать stack как геометрически регулярный 3D volume.
Изображение ROI min/max диагонали, floor/ceil по границе, resize bilinear;
окно 1–99 percentile всего исходного среза после rescale, scale[0,1].
Одинаковый preprocessing для всех классов; bbox/range подтверждать JSON.
Patient key namespace-qualified hash приватного идентификатора.
CSV codes должны совпадать с unionJSONcodes, пропуск/пустой JSON исключать.
Не смешивать normal0 с pathology. Path traversal/links/duplicates TAR запрещены.

## План

1. tools/prepare_tmj_od3d.py: локальный directory intake и безопасный streaming
   TAR download. Обрабатывать одного пациента за раз, приватный временный cache,
   сохранять originalbasename для reference binding. Писать prepared dataset
   atomically, не перезаписывать готовые artifacts и проверять hashes при resume.
2. Отдельный preprocessing helper training/tmj_od3d_images.py:
   prepare_patient(folder, metadata, output_root, slice_count=16,image_size=96)
   → records плюс excluded counters. Только native uncompressed DICOM,
   одна серия, совпадающие Rows/Columns/orientation/spacing; missing slices
   допустимы, не замещаются нулями. Проверять reference, SOP unique,
   координаты и frame dimensions. Не выводить private tags.
3. private index schema_version=1/task=tmj-osseous-author-roi-v1:
   source_doi,codebook_commit,preprocessing,records; record содержит patient_id,
   side,codes,binary_target,crop_path,crop_sha256,source_sha256,
   annotation_sha256,instance_numbers,roi_source=author_annotation,
   assessment_input=oracle_roi. Массив NPZ поле images имеет[K,1,H,W].
4. tests/test_tmj_od3d_images.py: sparse actual InstanceNumber,reference binding,
   normal/multiplecodes,CSVmismatch, bounds,duplicateinstances/SOP,
   croppedpixelhash mutation; real singlepatient smoke aggregateonly.
5. Проверить pilot и затем полный release streaming, данные внеGit.
   ПолныйTAR ~69GiBнехранится; rawcache≤1GiB,prepared≤5GiB,таймауты/retry.
   При interrupted HTTP resume от TAR границы завершённого пациента.

## Результаты

Не выполнено. Критерий завершения полного набора: достигнут EOF архива,
каждая CSVстрока имеет inclusion/exclusion reason, checksumindex и фактические
counts; небольшая smokeвыборка не является готовностью всей когорты.


### Ограниченный Range reader и восстановление

`tools/prepare_tmj_od3d.py::RangeReader` получает архив четырьмя HTTP Range
workers, chunk 8 MiB, максимум четыре queued chunks; отдаёт байты строго по
порядку и не сохраняет полный TAR. Вместе с текущим buffer и ограниченным
read output объём собственных byte buffers остаётся ниже 64 MiB. Каждый
ответ проверяет 206, точный Content-Range/Content-Length, identity encoding
и длину body; максимум три попытки, ошибки транспорта экспортируются
фиксированным кодом. Лимит download задаёт разрешённый диапазон уникальных
байтов, включая prefetch; повторы могут повторно перенести этот диапазон,
до трёх попыток каждого chunk. На patient limit/исключении pending futures
отменяются, активные workers закрывают ответы и join; timeout каждого запроса
30 секунд. Absolute TAR offsets для resume сохранены.

Поддерживаемый корень фактического V2 TAR — tmj. Запрещены traversal,
links/special files, вложенные подпапки и duplicate members, включая каталоги.
Размер prepared storage вычисляется при resume и затем только для новых
результатов завершённого пациента. Превышение 5 GiB удаляет новые
uncommitted side crops и останавливает запуск.

Lock содержит PID. Явный `--reconcile-stale-lock` отказывает при живом или
непроверяемом/старом пустом lock; PID должен отсутствовать по os.kill(pid,0).
Если lock отсутствует, recovery сначала приобретает собственный O_EXCL lock.
Строгие ours-format 64hex-L/R.npz вне сохранённого state переносятся в
приватный quarantine, общий quarantine limit 5 GiB; неизвестные файлы,
symlinks и invalid state отказывают без удаления данных. При обычном resume
orphans отклоняются и не превращаются в исключённых пациентов.

Новые patient_receipts сохраняются только в preparation.private.json:
namespace hash пациента, tar_start/tar_end, status/code, side_count.
Они исключены из training index и позволяют адресно повторить проблемный
диапазон без повторного сканирования полного TAR. Старый state получает
пустой receipt list; исторические диапазоны не восстановлены догадкой.

Проверка этого изменения: targeted
`cloud-pins-venv/bin/python -m pytest -p no:cacheprovider MLService/tests/test_prepare_tmj_od3d.py -q`
из корня — **25 passed**; TDD red наблюдался для отсутствующего reader,
нарушений layout/retry privacy, storage rollback и explicit orphan recovery.
Ruff F/I checks двух файлов — PASS; весь compact formatting не заявлен
проверенным. Live source/performance и полный набор этим шагом не проверялись;
действующий процесс подготовки не прерывался и не перезапускался агентом.

## Реальный release: отсутствующий референсный файл

Author render_sample_slice выбирает существующий InstanceNumber и применяет
bbox.contains_slice независимо от наличия image_path. Для slice-bag
референсный файл необязателен: ни один отсутствующий срез не создаётся.
Проверять folder/CSV/side/codes, общие frame dimensions и только опубликованные
срезы в объявленном диапазоне; число отсутствующих reference images сохранять
в provenance. Присутствующий reference вне range по-прежнему ошибка.
Старый проход остановлен по SIGINT после подтверждения live PID и завершения
handle; его артефакты сохранены. Новый полный проход нужен для восстановления
исследований, ошибочно исключённых строгим требованием reference filename.

Ревью обнаружило потерю валидной стороны при пустом JSON второй стороны.
Две регрессии (L/R) сначала падали с empty_annotation; исправлено локальное
исключение только пустой стороны с reason. Остальные ошибки аннотации
по-прежнему отклоняют пациента. Live процесс использует загруженную ранее
версию helper; новые файлы не меняют его поведение. После полного прохода
потребуется адресно проверить такие failure receipts при их наличии.

Полный проход остановился транспортной ошибкой source_range_failed после
77 завершённых пациентов. Exec handle подтвердил exit 1; raw-cache удалён
штатным finally, checkpoint и готовые crops сохранены. Возобновление выполняется
в тот же output root с проверкой hashes, от next_offset, без второго live reader.
Последний helper включает исправление empty_annotation; предыдущие prepared
records не менялись. Обнаружено два annotation_side_conflict, требующие 008.

## Диагностика transport rate limit

Однократный диагностический запрос 512 bytes к checkpoint offset подтвердил
HTTP 429 без Retry-After. Повтор source_range_failed не был ошибкой DICOM
или checkpoint. План: Range reader различает source_rate_limited и перед
повтором 429 учитывает ограниченную паузу (Retry-After с пределом 60 s,
без заголовка 30/60 s). Пауза interruptible через существующий stop Event,
число попыток остаётся три. Тесты без реальных пауз проверяют schedule и
закрытие; live full scan возобновляется только после завершения старого handle.

Backoff regression сначала наблюдался red в пяти сценариях. После исправления
все 30 streaming tests — PASS за 0.47 s; raw HTTP reason/URL не выводятся.
Подготовка, staging и side exclusion отдельно проверены: 57 focused tests — PASS
перед добавлением rate limit regression. Исходный HTTP 429 остаётся внешним
ограничением, исправление reader не выдаётся за завершение загрузки.

## Приёмка восстановления EOF

Независимое ревью обнаружило два окна сбоя: после последнего patient state
(next_offset=ARCHIVE_BYTES, complete=false) и между сохранением complete state
и index. План: при resume проверить coverage EOF checkpoint, завершить state
и восстановить derived index до возврата без сетевого чтения. Fault injection
на обеих границах должен воспроизвести дефект и подтвердить исправление.

Обе fault-injection регрессии сначала падали: invalid_download_budget и
устаревший complete=false index. После исправления все 32 targeted tests —
PASS за 0.70 s. Resume EOF не вызывает urlopen и сохраняет hashes crops.
После cooldown реальный supervisor возобновил source scan с 77 до 86
завершённых пациентов; это ещё не полная когорта.
