# Локализация и ROI

Результат локализации — две анатомически различимые области ВНЧС в определённой
системе координат. Исследовательские модели и HTTP-путь имеют разные реализации;
их карта находится в [ai/models](../../../ai/models/README.md).

<a id="fr-ml-loc-coordinates"></a>
### FR-ML-LOC-COORDINATES — Стороны и координаты

Центры и ROI должны сохранять left/right, порядок осей и связь с исходным объёмом. Для текущего локализационного обмена порядок — z,y,x, единица — воксель исходного массива. Paired heatmap path сохраняет InstanceNumber-ascending и исторический floor(index × original_shape / target_shape); физические позиции проверяются в этом порядке, без молчаливой IPP-перестановки. Отражение изображения должно согласованно преобразовывать стороны и координаты.
Приёмка: [TC-ML-LOC-COORDINATES](#tc-ml-loc-coordinates).

<a id="tc-ml-loc-coordinates"></a>
### TC-ML-LOC-COORDINATES — проверка

- Предусловие: заданы асимметричные центры и объём известной формы.
- Действие: применить преобразование и обратный перевод координат.
- Результат: каждая сторона соответствует исходной аннотации; оси и единицы не перепутаны.
- Покрытие: [ROI tests](../../../../tests/test_roi_provenance.py) проверяют InstanceNumber order, оси z,y,x, LPS→RAS и affine roundtrip поддержанной синтетической серии. Физическая ориентация всей цепочки остальных реализаций не проверена. [GAP-ML-PIPELINE-GEOMETRY](../../open-questions/README.md#gap-ml-pipeline-geometry).

<a id="fr-ml-loc-checkpoint-compatibility"></a>
### FR-ML-LOC-CHECKPOINT-COMPATIBILITY — Совместимость модели

Архитектура, число выходных каналов, preprocessing и способ извлечения центра должны соответствовать checkpoint. Регрессия, общая двухканальная heatmap и пара одноканальных heatmap не взаимозаменяемы без явного совместимого пути.
Приёмка: [TC-ML-LOC-CHECKPOINT-COMPATIBILITY](#tc-ml-loc-checkpoint-compatibility).

<a id="tc-ml-loc-checkpoint-compatibility"></a>
### TC-ML-LOC-CHECKPOINT-COMPATIBILITY — проверка

- Предусловие: доступны паспорта трёх семейств детектора.
- Действие: сопоставить веса с выбранным loader/evaluator.
- Результат: несовместимое семейство не объявлено поддержанным; конкретный путь определён.
- Покрытие: [ROI tests](../../../../tests/test_roi_provenance.py) загружают малый checkpoint paired-family и отказывают чужому семейству; совместимость реальных весов других путей не подтверждена. [ai/models](../../../ai/models/README.md).

<a id="fr-ml-loc-roi-geometry"></a>
### FR-ML-LOC-ROI-GEOMETRY — Семантика кропа

ROI должен включать заявленную область вокруг центра и сохранять связь с исходными voxels. Padding, обрезка у границы и отсутствие/ошибка стороны должны быть различимы в паспорте данных. Нельзя выдавать voxel affine за подтверждённую физическую геометрию.

Окно задаётся requested_start = center − size//2 и end = requested_start + size;
центр должен быть целым voxel внутри исходного массива. Пересечение с исходником
помещается по смещению относительно requested_start, padding выполняется отдельно
на каждой границе, в том числе для нечётного size. Поддержанная DICOM-серия —
согласованные monochrome single-frame slices, уникальный InstanceNumber, общие
размеры/идентичность серии и согласованные orientation/spacing/regular positions.
Непустые валидные StudyInstanceUID/SeriesInstanceUID обязательны и должны быть
общими; FrameOfReferenceUID либо отсутствует у всех, либо валиден и согласован.
UID проверяет coherent series, не patient identity; группировку задаёт canonical
вход. Частичная, mixed, tilted или irregular geometry отказывает. Если все physical tags
отсутствуют, допустим явно voxel-space; это не подтверждённые patient coordinates.

Для выбранного research path паспорт пары связывает source bytes в проверенном
порядке, source geometry, хеши обоих checkpoints и family, точные preprocessing
options/version, сторону, форму, checksum и requested crop transform. Missing,
corrupt или несовпадающий паспорт блокирует ready; skip-existing допускает только
полное совпадение, иначе обе стороны генерируются заново. Crop-only проверка
сохранённого паспорта явно сообщает отсутствие повторной проверки raw source.
Паспорт содержит только разрешённые geometry/scalars/hashes/counts и псевдонимный
study key; DICOM headers, patient values, UIDs, filenames и paths не копируются.
Техническая проверка не доказывает anatomical quality, анонимность или отсутствие
пациентов исследования среди обучающих данных детектора.

Поддержанный research профиль проверяется до pixel decode и padding: direct
single-series каталог, не более 4096 regular entries и 1024 slices, не более
1024 rows/columns и 512³ decoded source voxels, 64 MiB на source file и 1 GiB
на серию. Это допускает обычный 512³ объём без расширения профиля. Поддержаны
Part-10 native uncompressed single-frame integer PixelData (8/16/32 bits);
compressed syntax даёт `compressed_source_unsupported` до codec/allocation.
ROI — float32 NIfTI-1, crop edge 1–256, файл ≤128 MiB; паспорт ≤64 KiB.
Границы технические, не клинические; другое представление требует отдельного
обоснования поддержки, не обхода диагностики.

Инвентаризация direct entries не зависит от регистра/расширения: .dcm/.DCM,
extensionless и DICM-preamble files проходят строгий reader. Отсутствующий
Part-10 preamble не объявляется поддержанным. JSON/TXT/CSV без DICM явно
игнорируются как sidecars; неизвестные файлы, symlinks или подкаталоги отказывают,
не превращаются в молчаливо неполный объём. Паспорт хранит original physical
transform; фактический sform проверяется против его NIfTI-1 float32 serialization,
а не произвольно расширенной tolerance. Numeric fields имеют фиксированные
vector/matrix shapes без обхода arbitrary nested trees; отказ остаётся safe JSON.
Приёмка: [TC-ML-LOC-ROI-GEOMETRY](#tc-ml-loc-roi-geometry).

<a id="tc-ml-loc-roi-geometry"></a>
### TC-ML-LOC-ROI-GEOMETRY — проверка

- Предусловие: синтетические landmarks у всех границ, odd/oversized окно, валидная либо mixed DICOM-серия и пара ROI с паспортами.
- Действие: построить ROI, проверить affine/transform, изменить source/weights/options/checksum/side/shape/passport и повторить preflight/cache check.
- Результат: landmarks сохраняют исходное положение, форма и one-sided padding известны; invalid center и unsupported series отказывают; каждый mismatch блокирует ready. В voxel-space нет scanner affine; crop-only не выдаёт source_rechecked=true.
- Покрытие: [ROI tests](../../../../tests/test_roi_provenance.py), [intake CLI tests](../../../../tests/test_canonical_training_inputs.py): runtime synthetic DICOM→tiny paired detector→ROI→private strict index→preflight, без реальных снимков и обучения.

<a id="fr-ml-loc-localization-evaluation"></a>
### FR-ML-LOC-LOCALIZATION-EVALUATION — Оценка локализации

Ошибка центра сообщается отдельно от качества классификации, с единицами, составом выборки и способом извлечения центра. Перевод в миллиметры требует валидной геометрии конкретного исследования; независимость оцениваемых пациентов от обучения детектора проверяется для всей цепочки.
Приёмка: [TC-ML-LOC-LOCALIZATION-EVALUATION](#tc-ml-loc-localization-evaluation).

<a id="tc-ml-loc-localization-evaluation"></a>
### TC-ML-LOC-LOCALIZATION-EVALUATION — проверка

- Предусловие: есть predictions, аннотации, spacing и группы обучения.
- Действие: сформировать отчёт локализации.
- Результат: единицы и support явны; пересечения групп и неподтверждённый spacing отмечены.
- Покрытие: Общий evaluator поддерживает скалярный voxel-mm; patient independence не доказана. [GAP-ML-DETECTOR-LEAKAGE](../../open-questions/README.md#gap-ml-detector-leakage).

Проверка сжатия читает только Part-10 file meta до `dcmread`: deflated dataset тоже отвергается до распаковки. ROI имеют фиксированный NIfTI-1 header без расширений и текстовых полей, со смещением данных 352 байта; ограниченный header читается до nibabel. Gzip допускает один member с нулевым timestamp и без optional filename/comment/extra/header-CRC полей. Точное число float32 voxel bytes проверяется с ограничением распаковки; дополнительные members и хвостовые данные отвергаются. Это исключает перенос текстовых metadata и не доказывает анонимности пикселей.
