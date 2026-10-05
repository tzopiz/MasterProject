# Открытые CBCT/TMJ для собственной разметки положения

Первый кандидат — **TMJ-OD3D V2**. Наборы ниже содержат изображения и чужую
разметку, но **не готовые project position labels**. Ранжирование — предложение
для проверки доступной когорты, не подтверждение качества/пригодности обучения.
Primary-source сверка: 2026-10-02/03; изображения не скачивались, аккаунты не
создавались, соглашения не принимались. Public metadata не гарантирует доступ
к файлам или достаточное анатомическое покрытие каждого исследования.

| Приоритет | Подтверждённые сведения и прямой источник | Что проверить до выбора |
|---|---|---|
| **1. TMJ-OD3D V2** | [Карточка Science Data Bank / DOI](https://doi.org/10.57760/sciencedb.37727): PUBLIC, CC BY 4.0; V2 обновлён 2026-09-03. 1 043 CBCT-исследования от 1 043 пациентов, 2 086 left/right joints; TMJ-related axial DICOM slices, side-specific JSON с osseous abnormalities/lesion boxes и slice ranges. Два файла, 69.26 GB. | Coverage всего condyle **и fossa**, jaw state, series geometry/encoding, применимость проекта к исходной разметке. Диагноз/lesion box не определяет anterior/central/posterior. Заявленная de-identification не заменяет локальную проверку приватности; отдельность от собственной когорты и detector training неизвестна. Условия фактического доступа к файлам проверить перед pilot. |
| **2. ToothFairy2 — условно** | [Официальный портал](https://ditto.ing.unimore.it/toothfairy2/), [challenge dataset](https://toothfairy2.grand-challenge.org/dataset/), [benchmark dataset.json](https://github.com/AImageLab-zip/ToothFairy2-Benchmark/blob/main/dataset.json): CBCT, 480 training volumes, 50 test; CC BY-SA 4.0 в dataset metadata. Download требует аккаунт, private test не выдаётся. Benchmark указывает NIfTI `.nii.gz`; исторический портал — MHA и изменения geometry metadata, включая spacing 0.3 mm. | Зафиксировать **конкретный release/формат/spacing**, не смешивать описания портала и benchmark. Screen public train для bilateral TMJ/fossa; extended test FOV не доказывает такое покрытие train. Patient↔study/repeat mapping и размер пакета не подтверждены; masks dental structures не дают position labels. |
| **3. ToothFairy3 — версия пересекающейся когорты** | [Официальный портал](https://ditto.ing.unimore.it/toothfairy3/): 532 CBCT NIfTI volumes, 77 segmentation classes; A=417 и B=63 **повторяют TF2 volumes** с дополнительной разметкой. C=52 ранее не выпускались в TF2, иной scanner. RPI orientation; аккаунт для download. | Лицензия данных на открытой странице не установлена: не переносить TF2 license автоматически. Полное bilateral TMJ/fossa coverage, jaw state, patient mapping и bytes неизвестны. Broader FOV set B — не доказательство нужного покрытия. Новые volumes C не доказывают новых независимых пациентов. |
| **4. Bayrakdar et al. — restricted reserve / внешний test** | [Zenodo record](https://zenodo.org/records/20307507), [paper](https://onlinelibrary.wiley.com/doi/10.1111/joor.70247): v1 от 2026-05-20, карточка CC BY 4.0, **files restricted**. Original-study TEST images/GT/predictions: NIfTI condyle segmentation, cropped healthy/disorder subtype classification и grading; two-center/multidevice. | Access approval и дополнительные условия, counts/bytes, patient/group mapping, anatomy outside cropped condyle неизвестны. Crop может не содержать fossa. Это не доступный train cohort; при внешнем сравнении сохранить test independence и не использовать его для model/threshold selection. |

Причины не брать распространённые альтернативы «как есть»: original ToothFairy
и Maxillo — источники повторно использованных данных TF2; [портал TF2](https://ditto.ing.unimore.it/toothfairy2/)
явно описывает reuse и overlap Set A с ToothFairy. Они не увеличивают независимую
когорту без curated deduplication. Tooth/canal/condyle segmentation и disease
subtypes из любого shortlist не заменяют position targets. Private challenge test
и restricted Zenodo files не считать открытыми training inputs. Объёмы без
reference fossa или с непроверенной orientation исключать из position pilot,
даже если для исходной segmentation-задачи они подходят.

## Доступ, лицензия и производные материалы

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) разрешает sharing и
adaptation, включая коммерческие цели, с attribution, ссылкой на лицензию и
указанием изменений. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
добавляет same/compatible-license условие для распространяемых adaptations.
Эти условия позволяют планировать собственную разметку/исследовательскую ML-работу
на законно полученных материалах, но не доказывают автоматически право публиковать
любые weights, patient-level derivatives или сведения, защищённые другими правами.
Dataset license отделяется от article/code license; signup/access agreement
сверяется отдельно и выполняется владельцем доступа. Для TF3 и restricted набора
недостающие условия остаются блокером получения/использования данных, а не догадкой.

До pilot сохранить выбранные source DOI/release, текст лицензии/access conditions,
доступные provider patient/study mapping и source fingerprints **локально**.
TF2/TF3 используют один стабильный identity namespace для общей когорты:
release не становится новым source_id для обхода overlap. Повторные исследования
одного пациента и варианты разметки остаются в одной группе; совпадающие числовые
case IDs у разных источников не доказывают одного пациента. Межисточниковые
пересечения требуют curated mapping. ФИО, raw IDs/UIDs, приватные пути, исходные
изображения и patient-level labels/predictions не публикуются; хеш не гарантирует
анонимности. Public report — агрегаты и необходимые сведения об источниках.

## Небольшой clinician pilot после разрешённого доступа

1. Зафиксировать release и manifest; начать с небольшого заранее описанного
   patient-level набора TMJ-OD3D V2, например 10–20 уникальных пациентов. Схема
   отбора и exclusions записываются до просмотра labels; обе стороны и повторные
   серии учитываются вместе. Имеющийся технический probe первых трёх исследований
   не случайный и не representative; он не доказывает coverage или распределение
   position classes и не заменяет pilot.
2. Врач проверяет condyle/fossa coverage, стороны, ориентацию и возможность
   согласованных sagittal/frontal reformats, jaw state и image quality.
   Unknown/inconclusive сохраняются явно с reason, до обучения не допускаются.
   Для position-классов врач утверждает определения/annotation version и правила
   спорных случаев; числовые клинические thresholds здесь не выдумываются.
3. Размечать сначала обе sagittal стороны: project codes 1–3, затем при необходимости
   frontal 4–6. Статус нормы не следует из отсутствия osseous disease или label.
   Независимо повторно разметить заранее выбранную часть pilot; записать agreement,
   disagreements и adjudication. Итоговая версия относится к **конкретному study**,
   не автоматически ко всем сериям пациента.
4. Проверить source/geometry и [ROI паспорта](../spec/functional/localization/README.md),
   затем перенести только complete applicable records в
   [strict intake](../spec/functional/data/README.md#fr-ml-data-canonical-intake).
   Current writer поддерживает bounded native DICOM профиль; полный NIfTI/MHA
   из TF2/TF3 не превращается в supported raw input сменой расширения. Нужен
   явно подтверждённый source→ROI путь или уже проверенные ROI derivatives.
   Crop-only passport check не подтверждает повторную проверку отсутствующего raw source.
5. Решение о расширении когорты основывать на пригодной anatomy, наблюдаемом
   support классов, agreement и доступных patient mappings. Technical pipeline
   success не является доказательством медицинского качества или final-test результата.

## Предлагаемый приватный annotation ledger

Это шаблон ручной разметки, **не новый исполняемый intake schema**. Все примеры
synthetic; дополнительные сведения ledger остаются вне строгого schema-v1 JSON.

| Поле | Назначение / пример |
|---|---|
| `source_id`, `patient_id`, `study_id` | Supplied stable namespace/identity: `synthetic-source`, `patient-a`, `study-a`; curated mapping хранится отдельно приватно. |
| `side`, `plane` | `left`/`right`, `sagittal`/`frontal`, после проверки orientation. |
| `label_record_id`, `annotation_version` | `labels-a-v1`, `v1`; версия/источник врачебного решения. |
| `status`, `label_code`, `reason` | `unknown` + `null` до решения; после annotation/adjudication — допустимый project code. Reason: missing anatomy, unverified jaw state/geometry, inconclusive или excluded. |
| `label_applicability` | Подтверждение применимости записи именно к study; `confirmed` только после проверки. |
| `rater_key`, `adjudication` | Локальные псевдонимные rater IDs, disagreement и окончательное решение; не публичные patient-level результаты. |
| `source_release`, `source_reference`, `source_sha256`, `geometry_review` | Фиксированная версия/DOI, fingerprint и результат проверки geometry; provenance не подменяет identity. |

При экспорте в существующий strict JSON source/patient/label_record_id входят
в label row, source/study/patient/label_record_id и confirmed applicability —
в study row; пары side/plane становятся `labels.sagittal.left/right` и optional
`labels.frontal.left/right`. Unknown/null не преобразуются в code=1: такие study
не training-ready. Для sagittal-only frontal может отсутствовать. Ledger version
прослеживается через label_record_id и input hash, без добавления unsupported
полей в strict JSON. Правила кодов и преобразование binary central/non-central
переиспользуются из [canonical intake](../ai/data/README.md).

## Отдельный путь по авторской разметке костных изменений

[Аудит TMJ-OD3D от 2026-10-05](../../experiments/tmj_od3d_feasibility_20261005/README.md)
проверяет возможность обучения без новой разметки наших врачей. Это другая
целевая задача; текущий position runner для кодов этого источника непригоден.
В CSV есть пропуски и сочетания кодов; авторский codebook пока не подтверждён.
До его получения — no-go для обучения. Прежний probe трёх исследований
проверял геометрию; новая репрезентативная выборка и оценка анатомии не выполнены.
