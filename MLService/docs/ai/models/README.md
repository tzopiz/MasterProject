# Модели: несовместимые пути

| Семейство | Проверенный путь | Выход / извлечение |
|---|---|---|
| Регрессия | [tmj_detector](../../../models/tmj_detector.py), [detector service](../../../services/detector_service.py) | 6 sigmoid координат left zyx, right zyx; HTTP масштабирует по original shape |
| Общая heatmap | [tmj_heatmap_detector](../../../models/tmj_heatmap_detector.py), [trainer](../../../train_heatmap_detector.py) | По умолчанию 2 канала logits; [utilities](../../../training/utils/heatmap.py) используют softmax centroid |
| Пара heatmap | [crop generator](../../../tools/auto_crop_from_detector.py), [ноутбук](../../../google_colab/train_heatmap_detector.ipynb) | 2 отдельные одноканальные модели; argmax и поосевое масштабирование |

Общий evaluator не воспроизводит оценку пары моделей автоматически. Имена .pth
не задают совместимость. [Регрессионный dataset](../../../training/datasets/tmj_detector_dataset.py)
сортирует InstanceNumber и не применяет HU-rescale; HTTP применяет slope/intercept.
Общий [heatmap dataset](../../../training/datasets/tmj_heatmap_dataset.py) имеет cached
npy/255 и DICOM percentile paths; augmentation использует hardcoded factor 6.

Paired [crop generator](../../../tools/auto_crop_from_detector.py) сохраняет
InstanceNumber order, percentile 2/98 и scipy zoom order=1 до 96×128×128;
argmax переводится floor(index × original_shape / target_shape). ROI сохраняет
rescaled source float32 и requested origin, one-sided zero padding, включая odd size.
Это генерация входов, не patient-grouped split классификатора.

Основной путь — strict schema-v1 с supplied source/study/patient/label identity:

```bash
python tools/auto_crop_from_detector.py --canonical-input inputs.private.json \
  --dataset-root /mounted/research --output /mounted/research/roi \
  --left-model models/left.pth --right-model models/right.pth --crop-size 128 --skip-existing
python tools/check_training_inputs.py --input-json /mounted/research/roi/inputs.private.json \
  --dataset-root /mounted/research
```

Существующие `series_path` разрешаются strict intake; файл не требует `crops` до
генерации. Output ROI directories используют хеш source-qualified study key,
файлы — left/right.nii.gz. Writer добавляет relative crops к тому же strict JSON,
сохраняет labels/supplied IDs в **приватном** `inputs.private.json`; custom
`--index-output` требует суффикс `.private.json`. JSON не является публичным отчётом.
Historical split JSON/train+val+test/`--studies` доступны только с `--legacy-input`;
эта ветка не строит canonical label join.

[Shared helper](../../../training/roi_provenance.py) читает одну поддержанную
monochrome single-frame DICOM-серию, проверяет unique InstanceNumber, размеры,
valid nonempty Study/Series UID, orientation/spacing и regular orthogonal slice positions.
Frame UID может отсутствовать у всех; при наличии он должен быть валиден и общим.
Эта проверка не выводит patient identity из UID/PatientID.
InstanceNumber не заменяется IPP order. Physical affine имеет array axes z,y,x,
LPS→RAS и requested-start shift; NIfTI sform code=1, units mm. Полное отсутствие
physical tags допускает explicit voxel-space: transform в паспорте, sform/qform
code=0, units unknown. NIfTI `.affine` fallback в этом режиме нельзя считать
patient geometry. Partial/mixed/tilted/irregular series или decoder warnings
дают fixed-code отказ; raw diagnostics не печатаются. Для NIfTI-1 original affine
остаётся в паспорте, сравнение sform учитывает ровно serialized float32 header,
включая fractional origin; увеличение arbitrary tolerance не применяется.

Один поддержанный профиль проверяется до decode/allocation: ≤4096 direct entries,
≤1024 slices, ≤1024 rows/columns, ≤512³ decoded voxels, source file ≤64 MiB,
source bytes ≤1 GiB; crop edge ≤256, float32 NIfTI-1 ≤128 MiB, passport ≤64 KiB.
512³ проходит limit-check без allocation в regression. Native uncompressed
8/16/32-bit integer Part-10 DICOM поддержан; compressed transfer syntax даёт
`compressed_source_unsupported` до codec invocation. Иные данные сейчас явно
неподдержаны, а не автоматически перекодируются. Это технические ресурсные
границы, не пределы клинической пригодности.

Inventory ограничен direct files выбранной серии, без recursive scan. .dcm/.DCM,
extensionless и DICM-preamble files читаются строго; JSON/TXT/CSV без DICM
игнорируются как известные sidecars. Unknown files, subdirectories и symlinks
дают отказ. Нельзя молча использовать только lowercase .dcm подмножество.
Byte budget ограничивает чтение; metadata UID/shape/syntax/geometry checks
выполняются до pixel_array, затем native PixelData length проверяется перед
декодированием. Source float32 volume выделяется один раз; planes не дублируются
через stack. Fixed-shape vector/matrix checks и bounded passport bytes исключают
рекурсивный обход произвольных вложенных значений. CLI help сообщает этот профиль.

`write_roi_crop(volume, center, crop_size, path, *, side, record, source, detectors)`
пишет NIfTI и `str(path)+'.passport.json'`. Паспорта всех уровней whitelisted:
source fingerprint/geometry, оба checkpoint hashes и paired family, preprocessing
version/options (включая размер), side, checksum/shape/center/requested bounds,
padding и affine. Никаких copied headers, UIDs, PatientName/ID, filenames или paths.
Study key — псевдоним, не гарантия анонимности. Новые NIfTI не наследуют DICOM headers.

`validate_roi_pair(record, *, expected_detectors=None, expected_preprocessing=None,
recheck_source=True)` используется CLI и research runner; он проверяет genuine
NIfTI, обе стороны и паспорт. `ROIValidationError.code` — безопасная причина отказа.
Если raw series доступна, fingerprint/geometry вычисляются повторно; crop-only
отчёт явно содержит source_rechecked=false. `--skip-existing` сравнивает также
оба текущих checkpoint hashes и preprocessing_options(crop_size); отсутствующий,
stale, corrupt либо one-sided cache пересоздаёт обе стороны.

Loader `load_paired_detector(path, device)` поддерживает прежние state_dict с
архитектурой [32,64,128,256]; optional `detector_family` должен совпадать, а
`model_config.features` позволяет загрузить малую ту же архитектуру для synthetic
CPU checks. `prepare_input`, `argmax_to_orig`, `load_validated_series` и
`generate_roi_pair` доступны следующему CLI без второго reader/preprocessor.
Техническая family/hash проверка не подтверждает side-specific anatomical quality
или independence от detector training cohort; detector_training_identity=unknown.
[ROI tests](../../../tests/test_roi_provenance.py) проверяют эту ограниченную область,
не совместимость прежних кропов с Drive и не качество настоящей модели.

[2D dataset](../../../training/datasets/tmj_dataset.py) использует HU clip [-1000,2000]
и mask>0.5; [3D dataset](../../../training/datasets/tmj_3d_dataset.py) — min/max и mask>0.
[Evaluator](../../../validation/evaluate_model.py) выполняет slice/max, resize и output>0.5.
В [metrics](../../../validation/metrics.py) Hausdorff в voxels, ASD принимает spacing.
Это текущие технические пороги, а не утверждённые пороги научной приёмки.

[Heatmap model tests](../../../tests/test_tmj_heatmap_detector.py),
[dataset tests](../../../tests/test_tmj_heatmap_dataset.py) и
[loss tests](../../../tests/test_heatmap_loss.py) существуют; здесь не запускались.


## Shared per-side classifier checkpoint

[ROI classifier](../../../models/tmj_binary_position_classifier.py) сохраняет
существующий binary backbone/default two single-logit heads. num_classes=3
создаёт только sagittal three-logit head; output (sagittal[B,3],None), frontal
head отсутствует. Legacy [four-head whole-volume model](../../../models/tmj_position_classifier.py)
не меняется и не является canonical multiclass ROI path.

`load_position_checkpoint(path,device="cpu",expected_sha256=None,expected_mode=None)`
возвращает eval model и metadata; normalized mode/num_classes присутствуют и для
старого binary schema1. `load_binary_position_checkpoint` остаётся binary-only,
принимает старый contract и отказывает multiclass. Generic fixed refusal:
invalid_position_checkpoint; binary compatibility refusal: invalid_binary_checkpoint.

Binary family tmj_binary_position_classifier сохраняет threshold и >=. Multiclass
family tmj_roi_sagittal_multiclass_classifier/schema1 явно задаёт mode=multiclass,
num_classes=3, classes0central/1anterior/2posterior, trained_tasks=[sagittal],
threshold=null и rule argmax;ties-lowest-class-index. Model kwargs/state shape,
finite weights, preprocessing и optional report checksum проверяются weights_only
loader; wrong mode/family/count/semantics/corruption отказывают безопасно.

Preprocessing одинаков: NIfTI array float32, percentile2/98, per-volume0..1,
constant-volumezero, NCDHW/onechannel/noresampling. Его существующий owner —
`training/datasets/tmj_position_dataset.py::_normalize_volume_percentile`; dataset
добавляет channel и batch без transpose. Version string tmj-binary-nifti-percentile-v1
сохранён также для multiclass, поскольку входные преобразования совпадают.
Checkpoint локален; private rows/replay не публикуются. Проверки:
[multiclass actualCPU train/reload](../../../tests/test_multiclass_research.py) и
[binary artifacts](../../../tests/test_cv_artifacts.py). Качество реальных весов
или совместимость реально переданной когорты из этих тестов не выводятся.
