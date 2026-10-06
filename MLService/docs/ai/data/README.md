# Данные: реализация исследовательского intake

Основной вход — schema-v1 JSON и явно выбранный локальный/смонтированный root.
[Label table](../../../training/tmj_position_label_table.py) предоставляет
`build_canonical_index(input_path, dataset_root, *, sagittal_only=True, require_crops=True)`.
Возвращаются записи в устойчивом порядке `(source_id, study_id)`; исходные ID
сохраняются. Неверный root не заменяется другим каталогом.

## Минимальный JSON

```json
{
  "schema_version": 1,
  "studies": [{
    "source_id": "cohort-a",
    "study_id": "scan-a",
    "patient_id": "p-a",
    "label_record_id": "labels-v1-a",
    "label_applicability": "confirmed",
    "series_path": "dataset_public/scan-a",
    "crops": {
      "left": "roi/scan-a/scan-a_left.nii.gz",
      "right": "roi/scan-a/scan-a_right.nii.gz"
    }
  }],
  "labels": [{
    "source_id": "cohort-a",
    "label_record_id": "labels-v1-a",
    "patient_id": "p-a",
    "labels": {"sagittal": {"left": 2, "right": 1}}
  }]
}
```

ID — непустые opaque tokens `[A-Za-z0-9][A-Za-z0-9_.:-]*`, предоставленные
владельцем данных, а не восстановленные из номера случая/серии, имени или
DICOM PatientID. Дополнительные медицинские сведения и приватное соответствие
остаются вне этого компактного входа. Opaque ID и hash — псевдонимизация,
а не гарантия анонимности. Канонический индекс, полные labels и приватный mapping
хранятся локально/в разрешённом приватном хранилище, не в публичном отчёте или Git.
CLI минимизирует отчёт до counts и codes даже при чувствительном supplied ID.
`label_record_id` должен различать версии
разметки. Один файл может содержать несколько источников.

`source_id` — устойчивое пространство идентичности исходной когорты, **не номер
выпуска/выгрузки**. Пересекающиеся releases одной когорты используют тот же
source_id. Разные источники с известными общими пациентами требуют проверенного
канонического mapping перед объединением: автоматически обнаружить эту связь
валидатор не может. Синтетический успех не подтверждает mapping реальных данных.

Join использует `(source_id, label_record_id)` и проверяет совпадение patient_id.
Каждая study-row отдельно подтверждает применимость метки; повторная серия не
получает её автоматически. Дубликат study/label key, конфликт, invalid type,
boolean/дробный/wrong-plane code, missing sagittal и unknown schema fields
отклоняют весь вход. Валидные label rows вне выбранных studies не добавляют
учебных примеров. Frontal опциональна только в sagittal-only режиме; если она
предоставлена, проверяются обе стороны и коды 4–6. `--all-planes` требует её.

Пути относительны dataset root. Absolute/escaping/symlink-escaping paths,
отсутствующий directory/file и повторное использование одного пути отклоняются.
Для готовых кропов series_path можно опустить. Для подготовки без кропов он
обязателен. Явно заданный неправильный series_path отклоняется и при наличии ROI.
Проверка наличия кропов **не проверяет NIfTI, геометрию и паспорт ROI**.

## API и команда

Canonical records содержат source_id, patient_id, patient_key, study_id,
label_record_id, class-valued sag_left/sag_right; fr_left/fr_right и dicom_dir
присутствуют только при соответствующих входах. crop_paths содержит проверенные
абсолютные пути. patient_key — JSON-пара `[source_id, patient_id]`; group helpers
формируют этот ключ из supplied полей. `binarize_labels()` сохраняет identity и
использует explicit crop_paths, проверяет классы 0–2, не подставляет missing fr.

```bash
python tools/check_training_inputs.py --input-json inputs.json --dataset-root /mounted/tmj
python tools/check_training_inputs.py --input-json inputs.json --dataset-root /mounted/tmj --allow-uncropped
```

CLI без загрузки модели/обучения выдаёт JSON и exit 0 при валидном выбранном stage,
exit 2 при отказе. Error report содержит только фиксированные codes, collection,
row number и field; исходные values, имена и paths не выводятся. API бросает
`InputValidationError` с тем же `.report`. Успех сообщает counts, `intake_ready`
и pending_checks. Основной stage=preflight вызывает shared
[ROI validator](../../../training/roi_provenance.py): genuine NIfTI + paired
passport обязательны, `training_ready=true` означает техническую готовность этих
входов. Aggregate provenance отдельно сообщает source-rechecked/cached-only
counts, coordinate spaces и unknown detector-training identity; качество модели,
применимость меток и анонимность не выводятся из ready. `--allow-uncropped` имеет
stage=intake-only, ready/training_ready=false и pending ROI generation/provenance,
даже при валидном intake. Intake API сохраняет file-only семантику.

## Исторический путь

`build_index()` остаётся явно legacy name join: stripped patient_name == name_raw,
дубликат name заменяет прежнюю строку, unmatched study пропускается. Legacy split
использует patient_name. Этот API не подтверждает patient identity и не является
строгим исследовательским preflight. Optional legacy cache содержит имена и остаётся приватным локальным артефактом;
legacy logs не выводят имена, study values или cache paths.

[Организатор](../../../tools/organize_dataset.py) назначает study/subject ID каждой
серии; [PHI-strip](../../../tools/dicom_phi_strip.py) заменяет PatientID на study ID.
Это не patient grouping. [Анонимизатор](../../../tools/anonymize_labels.py)
хеширует имена, но не доказывает однозначность личности. Legacy DOCX aliases,
fuzzy downloader и inline notebook join не участвуют в canonical intake.

## Проверки и ограничения

[Canonical tests](../../../tests/test_canonical_training_inputs.py) проверяют
источники/группы, repeat studies, стабильность порядка, join/type/label/path
отказы и CLI privacy. Existing [label-table tests](../../../tests/test_tmj_position_label_table.py),
[binary tests](../../../tests/test_binarize_labels.py) и
[group tests](../../../tests/test_stratified_group_kfold.py) сохраняют legacy regression coverage.
Фактические команды/результаты #86 приводятся в PR; локальные test artifacts
не публикуются как входные данные.
Правила — [данные](../../spec/functional/data/README.md).

Проверка ROI passport выполняется общим helper в preflight и выбранном research runner;
готовность размеров/групп и протокола runner проверяется самим runner. Реальные identity, применимость врачебных меток и межисточниковые
пересечения остаются ответственностью передаваемого mapping.

## Независимый osseous intake

Опубликованная разметка TMJ-OD3D обрабатывается отдельно: [контракт](../../spec/functional/osseous/README.md),
[CSV/JSON parser](../../../training/tmj_od3d_inputs.py),
[подготовка ROI](../../../training/tmj_od3d_images.py). Это не schema-v1 position
intake и не проектные коды 1–6 положения. Норма явно 0; сочетания патологий
сохраняются, пропуск исключается. В cloud передаются только подготовленные
NPZ и приватный index, без DICOM headers, sex/age или исходных имён.
Patient-level hashes, метки и predictions всё равно остаются приватными.
