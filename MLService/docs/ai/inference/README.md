# Исследовательский прогноз из выбранной DICOM-серии

[CLI infer_research.py](../../../tools/infer_research.py) реализует [FR/TC-ML-READINESS-INFERENCE](../../spec/features/dataset-readiness/README.md#fr-ml-readiness-inference). Вход — одна явно выбранная серия, две paired heatmap модели и каталог **завершённого** canonical CV run нужного режима. Это локальный research output по сагиттальному положению каждой стороны; clinical suitability и независимая test-оценка не установлены. Backend/iOS здесь не подключаются.

```bash
cd MLService
python tools/infer_research.py \
  --dicom-series data/selected-series \
  --left-model data/weights/left.pth --right-model data/weights/right.pth \
  --run data/completed-run --mode binary --output data/inference-run
```

Для трёх классов использовать `--mode multiclass` и соответствующий completed run. CPU — default; `--device cuda` или `mps` выбираются явно при доступном runtime. Новый output directory обязателен. Существующий каталог не перезаписывается и не переиспользуется после ошибки. CLI выдаёт только schema/status/mode/side_count/fold_count либо фиксированный код ошибки; решения и медицинские строки не выводятся в stdout.

## Saved run и переносимый контракт

Используются **все** fold checkpoints из `report.json`, без выбора лучшего validation fold. Проверяются complete/partial status, точное число folds и индексы 0..n−1, mode/classes, basename refs без symlink/path escape, checksums replay/checkpoints/private predictions, метаданные fold/best epoch/architecture/сохранённого binary threshold. [Общий loader](../../../models/tmj_binary_position_classifier.py) проверяет version/family/state shapes, число классов, trained sagittal task и classifier preprocessing.

Canonical [CV](../../../training/sagittal_binary_cv.py) и [research preflight](../../../tools/run_research.py) требуют одну общую комбинацию paired detector hashes/family и точных ROI preprocessing options у всей когорты. Guard [require_shared_roi_contract](../../../training/roi_provenance.py) одинаковый до model/output и до cloud prepare. Training сохраняет только compact `roi_contract={detectors,preprocessing}` в `report.provenance` и в hash-bound `replay.private.json`. Inference сверяет обе копии, а переданные веса — с обоими detector hashes; crop_size выбирается из этого контракта, не из нового default. Checkpoint schema классификатора не изменяется.

Это позволяет использовать скачанный completed run без исходного training dataset: приватные crop paths внутри replay не открываются. Исторические/legacy runs без generation contract, неоднородные cohorts, повреждённые или неполные артефакты получают отказ; inferred provenance или автоматического исправления нет. SHA-256 обнаруживает несогласованность сохранённых файлов, но не является цифровой подписью доверенного издателя.

Профиль reader: metadata JSON ≤16 MiB, отдельный artifact/detector checkpoint ≤256 MiB, суммарные referenced run artifacts ≤1 GiB, 2..64 folds. Все refs проверяются до построения classifier ensemble. Architecture profile общий для classifier/detector loaders, constructors и training preflight: один входной канал, 1..4 encoder blocks с 1..256 каналами; classifier fc_hidden 1..2048. Внутренний detector bottleneck допускает 512 каналов. Проверенный crop edge обязан быть ≥2**число classifier stages для MaxPool3d(2); CV/preflight отказывают до model/output, inference проверяет сохранённый ROI/replay contract до ensemble construction. Параметры вне профиля отклоняются до constructor/Conv3d allocation даже в малом checkpoint; defaults и tiny модели поддерживаются. Это ограничивает размеры параметров, но не обещает достаточную память для training activations, CUDA или любого batch/crop. Checkpoint читает только weights_only. Это поддерживаемый research profile, не обещание загрузить любое семейство/размер модели.

## DICOM, ROI и preprocessing

[generate_roi_pair](../../../tools/auto_crop_from_detector.py) использует существующие native DICOM reader, paired detector loader, историческое InstanceNumber ordering, detector resize/coordinate mapping и fixed-window padding. [ROI validator](../../../training/roi_provenance.py) проверяет source geometry, сопоставимость paired passports, affine/shape/checksums и generation options. Поддерживается ограниченный [DICOM/ROI профиль](../../spec/functional/localization/README.md), включая отказ compressed syntax до decoder expansion; Study/Series UID проверяют принадлежность серии, не patient mapping или диагноз.

Новые изображения имеют whitelisted technical passports без исходных UID/имён/путей и новый opaque study key. Проверенный физический transform сохраняется; отсутствие необходимой геометрии обозначается voxel-space. Array NIfTI читается без transpose/resampling. Нормализация переиспользует `_normalize_volume_percentile` из [того же dataset](../../../training/datasets/tmj_position_dataset.py): clipping2/98, per-volume0..1, constant→zero, float32. Tensor имеет NCDHW с одним каналом. Classifier preprocessing берётся из проверенных metadata; frontal head не используется и не выводится.

## Решение ансамбля и приватные файлы

Binary классы: 0=`central`, 1=`non-central`. Каждый fold применяет собственный сохранённый threshold по `>=`; majority решает класс, при равенстве — central. `vote_fraction` — доля non-central голосов, **не калиброванная вероятность**. Список fold decisions позволяет проверить правило.

Multiclass классы: 0=`central`, 1=`anterior`, 2=`posterior`. Усредняются softmax class probabilities всех folds; argmax выбирает класс, tie — меньший индекс. Эти model probabilities не выдаются за калиброванную уверенность или клинический риск. Ensemble не получает независимую test-оценку из той же development CV.

`prediction.private.json` содержит mode/classes, две стороны/решения, правило и технические hashes/preprocessing/provenance. `left/right.nii.gz` и паспорта остаются приватными. Directory mode0700, prediction/failure JSON0600; всё хранить под игнорируемым `data/`. Source hash, opaque study key и псевдонимизация не гарантируют анонимность pixel/face content. Detector training identity/independence остаются unknown. При ошибке после создания output сохраняются partial ROI и `failure.private.json` с фиксированным статусом; successful prediction не создаётся.

## Проверки

[Сценарии inference](../../../tests/test_infer_research.py) проводят настоящий tiny CPU train binary и multiclass, затем удаляют training dataset и выполняют synthetic DICOM→tiny paired heatmap→ROI→все saved folds→две стороны. Классификатор не mock. Проверяются точные thresholds/tie rules, совпадение нормализации, CLI privacy/no overwrite, ошибки серии/run/metadata/family/preprocessing/hash/path escape и mixed generation contract refusal уже в preflight. [ROI проверки](../../../tests/test_roi_provenance.py) отдельно контролируют ранние native/NIfTI header guards. Эти синтетические связи не доказывают anatomical quality, клиническую точность, независимость от detector-training cohort или работоспособность произвольного codec/GPU.

## HTTP: отдельный регрессионный путь

[app.py](../../../app.py) обслуживает регрессионную локализацию:
POST /process получает обязательный multipart task_id:string и повторяющиеся files.
UUID не проверяется. Файлы пишутся в temp_dir, cleanup выполняется в finally.
Внутренние исключения, включая локальный HTTPException, возвращают failed JSON
с error_message и обычным HTTP 200; входная валидация FastAPI находится снаружи.

Успешный ответ: task_id, status=completed, tmj.left/right как JSON objects,
center=[z,y,x], bbox=[z1,y1,x1,z2,y2,x2], volume_shape=[D,H,W]. Единицы — voxels
исходного массива; spacing/orientation/version/confidence в ответе нет.
List DTO не ограничивают длины и конечность чисел.

[DICOM loader](../../../services/dicom_processor.py) читает непосредственные *.dcm,
сортирует ImagePositionPatient[2], при AttributeError — filename. Slope/intercept
применяются; min/max всего объёма даёт uint8 [0,255], constant volume→zeros.
Проверки принадлежности одной серии и полной orientation нет.

[Detector service](../../../services/detector_service.py) resize до 96×128×128,
формирует [1,1,D,H,W], переводит 6 координат в original shape. Номинальный bbox
64 voxels обрезается границами. Startup: MODEL_PATH, лексикографически последний
experiments/detector_* с best_model.pth, затем fallback. config:model_type задаёт
архитектуру (default large), loader принимает model_state_dict или прямой state dict.
Heatmap checkpoint этот путь не поддерживает.

GET /health: status=ok и model_loaded; GET /models/status: model_loaded,
model_type=tmj_detector_3d, optional model_path (может быть строкой loaded).
Наличие этих маршрутов сверено статически; requests и model loading не выполнялись.
