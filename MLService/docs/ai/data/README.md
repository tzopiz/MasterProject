# Данные: проверенные особенности реализации

[Label table](../../../training/tmj_position_label_table.py) связывает
manifest.studies.patient_name.strip() с labels.patients.name_raw.strip().
Дубликат name_raw заменяет предыдущую запись в dict; unmatched study пропускается.
Неполная метка не становится central: доступ к обязательному полю завершится ошибкой.
Mapping проверяет 1–3 для sagittal и 4–6 для frontal; схемы JSON/legend целиком нет.

Индекс содержит study_id, dicom_dir, patient_name и четыре класса. Optional cache
сохраняет patient_name. Бинаризация даёт left/right crop paths; наличие NIfTI не
проверяется. Она считает любое nonzero значение non-central, поэтому безопасна
только после проверки исходных классов. Split группирует по patient_name.

[Организатор](../../../tools/organize_dataset.py) создаёт study/subject ID на серию.
[PHI-strip](../../../tools/dicom_phi_strip.py) меняет UID и PatientID на study ID.
Эти ID не объединяют серии пациента. [Анонимизатор](../../../tools/anonymize_labels.py)
заменяет stripped names SHA-префиксами, сохраняя структуру ключей для join;
он не исправляет неоднозначность идентичности и не доказывает полное обезличивание.

[Тест label table](../../../tests/test_tmj_position_label_table.py) покрывает basic
join/mapping/split; [тест бинаризации](../../../tests/test_binarize_labels.py) — записи
и пути. Тесты существуют, здесь не запускались. Требования:
[Требования данных](../../spec/functional/data/README.md).
