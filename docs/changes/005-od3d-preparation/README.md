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

Helper подготовки ROI реализован; 12 targeted tests проверены (включая
пустую сторону, sparse Instances, отсутствие reference и изменённые headers).
Три реальных технических пациента дали шесть приватных side bags;
это не репрезентативная оценка анатомии или качества модели.
Streaming/resume реализуется отдельным зависимым изменением этой задачи.
Критерий завершения полного набора: достигнут EOF архива,
каждая CSVстрока имеет inclusion/exclusion reason, checksumindex и фактические
counts; небольшая smokeвыборка не является готовностью всей когорты.


