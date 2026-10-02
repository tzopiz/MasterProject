# Оценка: что означает текущий отчёт

[Сагиттальная CV](../../../training/sagittal_binary_cv.py) принимает строгий
`input_path` через canonical index: patient groups квалифицированы `source_id`.
Имя используется только при явном `legacy_name_join=True`. Страта пациента —
max sagittal binary label всех его сторон/серий; асимметричный пациент относится
к положительной страте. Реальную идентичность и применимость разметки задаёт
входной файл, а не алгоритм разбиения.

До обучения проверяются параметры, число групп, наличие обоих классов в каждом
train/validation, отсутствие пересечений и полнота фолдов. Пустой, одноклассовый
или непригодный набор не становится успешным CV. Loss обучает sagittal head;
best epoch, scheduler и early stopping используют val AUC того же фолда.
Отчёт явно сообщает `assessment=development_cv`,
`model_selection_source=validation`; отдельного test нет.

Youden threshold рассчитывается на отдельном unaugmented/unshuffled loader
training records (`calibration_source=training_unaugmented`). Это training
calibration, а не независимая калибровочная когорта. Кандидаты ROC ограничены
конечными thresholds; при равном J выбирается наибольший конечный порог.
Старый [holdout trainer](../../../train_binary_position_classifier.py) выбирает
mean val accuracy и калибрует thresholds на той же val; он не заменяет canonical CV.

[Binary metrics](../../../training/utils/binary_metrics.py) применяют score>=threshold,
confusion order [0,1], explicit positive=1, support классов, sensitivity,
specificity и `f1_positive`. `f1_minority` сохранён только как legacy alias
positive-F1. При отсутствии класса его recall и balanced accuracy undefined;
AUC=NaN, Youden=0.5. Пустые/некорректные arrays отвергаются.
Train-majority baseline выбирает класс только по train (при равенстве — central),
оценивает те же val records и хранится рядом с метриками модели.

Fold seed задаётся до loader/model construction. [Seed helper](../../../training/utils/seed.py)
фиксирует Python/NumPy/Torch RNG, отключает cuDNN benchmark и включает deterministic
cuDNN convolutions; worker initializer сериализуется для spawn. Это не гарантия
битовой идентичности между CPU/MPS/CUDA, аппаратурой и версиями библиотек; прочие
недетерминированные kernels не запрещаются через глобальный strict режим.

CV summary считает mean/population std; это не доверительный интервал.
JSON/JSONL используют strict JSON (nonfinite metrics→null); `report.json`
атомарно заменяется до обучения и после каждого fold. `complete` означает все
запланированные folds, а не целевое качество. `failed`/`interrupted` сохраняют
completed folds и epoch trail; принудительное завершение процесса/сбой диска
может оставить последний `in_progress` snapshot. Автоматического resume нет.

## Локальные артефакты CV

Каждый запуск создаёт новый `output_dir/<opaque UUID>/` (по умолчанию
`experiments/`); `run_id` может явно выбрать локальный каталог, повторное
использование запрещено. Возвращаемый `run_key` совпадает с автоматически
созданным UUID; пользовательское имя каталога в публичный report не попадает.
Совместимость `output_json` временно сохранена как alias: `parent/stem`
выбирает каталог, внутри него пишется `report.json`. Отдельный JSON по старому
пути не создаётся; существующий файл или каталог означает отказ до записи.
`run_id` и `output_json` вместе запрещены.

`report.json` содержит allowlist hyperparameters, агрегированные метрики/counts,
`development_cv`, `independent_test=false`, safe provenance и ссылки на общие
имена артефактов с SHA-256. Canonical studies проходят общий
[ROI validator](../../../training/roi_provenance.py) до binarize/создания модели:
проверяется пара паспортов, checksums, geometry и preprocessing. До model/output также проверяется единый
ROI generation contract по detector hashes/family и точным preprocessing options;
тот же guard работает в research preflight до cloud prepare. Compact contract
сохраняется в report.provenance и hash-bound private replay для
[переносимого inference](../inference/README.md) без training paths. Отчёт отдельно
считает source rechecks и coordinate spaces. Crop-only подтверждает паспорта,
но не пересчитывает исходную DICOM-серию. Detector training identity и
independence остаются `unknown`; это не независимая итоговая оценка.

Каждый выбранный fold сохраняет `fold_XX_checkpoint.pth` с CPU tensors,
architecture args, `trained_tasks=[sagittal]`, class semantics, preprocessing
contract, best epoch, точным неокруглённым threshold и правилом `>=`.
Frontal head структурно существует, но не обучен. Загрузка:
[load_binary_position_checkpoint](../../../models/tmj_binary_position_classifier.py)
использует `weights_only=True`, проверяет family/version/контракт и strict state
shapes; `expected_sha256` из report обнаруживает подмену/битовое повреждение.
Ошибка возвращает только `invalid_binary_checkpoint` без пути или traceback
исходного загрузчика. CUDA/MPS→CPU predictions могут слегка различаться;
битовая идентичность между аппаратурой/версиями не обещается.

**Приватные материалы:** `replay.private.json` содержит полный config, локальные
пути, records, исходные mappings и SHA входов/кропов/паспортов;
`fold_XX_predictions.private.jsonl` содержит aligned labels/logits/probabilities,
threshold/decision, стороны, fold membership и SHA group/sample/crop.
Идентификаторы строк псевдонимизированы domain-separated SHA-256, это не полная
анонимность. Оба вида файлов остаются непубликуемыми patient-level данными;
приватный run directory создаётся с mode 0700, private файлы — 0600.
Медицинские объёмы в артефакты не копируются. Checkpoints остаются локальными:
создание прогонов не означает разрешения публиковать веса или outputs.

Input fingerprint для canonical — SHA-256 точных bytes входного файла; для
legacy — digest списка SHA manifest/labels. Dataset fingerprint — digest
упорядоченных sample/group keys, labels и crop/passport checksums. Публичный
report содержит только агрегированные digests, версии Python/packages,
эффективный device type, CUDA/cuDNN сведения и hashes затронутых исходников.
Имена пациентов, source/study IDs, per-crop hashes и приватные paths в него
не включаются. Эти fingerprints доказывают совпадение входов, не анонимность
или медицинскую пригодность.

`recompute_fold_metrics(private_rows)` восстанавливает train Youden, train-majority
baseline и все выбранные val metrics только по сохранённым predictions.
Summary пересчитывается тем же mean/population std поверх fold metrics.
Сбой внутри выполнения заменяет report на `failed` с фиксированным кодом,
сохраняет предыдущие артефакты и не публикует исходное сообщение исключения.
Новый прогон выполняется в новом каталоге с seed/config; продолжения старого нет.

Проверки: [CV regressions](../../../tests/test_sagittal_binary_cv.py),
[seed/spawn serialization](../../../tests/test_seed.py),
[metrics](../../../tests/test_binary_metrics.py),
[train/save/reload и privacy](../../../tests/test_cv_artifacts.py). Малые реальные CPU-folds и NIfTI
проверяют техническое выполнение и повторимость калибровки, не генерализацию.

[Multiclass trainer](../../../train_tmj_position_classifier.py) — whole-volume
четырёхголовый baseline, сумма четырёх CE; checkpoint по mean val accuracy.
[2D trainer](../../../train.py) делит volumes, [3D](../../../train_3d.py) — crops через
random_split; на малом наборе 3D допускает train=val.
Существующие [fold tests](../../../tests/test_stratified_group_kfold.py) и
[metrics tests](../../../tests/test_binary_metrics.py) не проверяют генерализацию.


## Безопасный CLI и экспорт старых отчётов

CV CLI возвращает JSON `ready=false, code=invalid_arguments` и exit 2 при
неверном argparse argument, features conversion или device specification;
исходные значения аргументов в диагностику не копируются. `--help` сохраняется.

Analyzer применяет тот же typed config allowlist к текущему и historical report.
До stdout/text/JSON/CSV/plots отбрасываются unknown/private поля configuration,
summary, fold/baseline и epoch rows; метрики/support/confusion имеют numeric types,
а assessment/status/selection/calibration labels ограничены известными enums.
Legacy positive-F1 aliases сохраняются. Неверный известный aggregate payload
отказывает с `invalid_cv_report` до вывода/записи; произвольный текст не превращается
в метрику или подпись графика. Полный replay config остаётся приватным.
Проверки synthetic secrets и historical payload — в CV regressions; они не
подтверждают медицинское качество модели.


## Shared sagittal multiclass development CV

Тот же [CV owner](../../../training/sagittal_binary_cv.py) и [runner](../../../tools/run_research.py)
принимают mode=multiclass на проверенных per-side ROI. Loss — cross-entropy,
selection — validation macro F1 (scores within1e-8 сохраняют раннюю эпоху),
scheduler/early stopping используют эту же metric. Пространственная augmentation
отказывает; default none. Training evaluator неаугментирован/неперемешан, но
multiclass threshold не калибруется: calibration_source=not_applicable.

Все train/val folds содержат три класса и disjoint patient groups. Group-stratum
heuristic — maximum sagittal class index пациента, с явным отказом неподходящего
split. Per-class precision/recall/F1/support и confusion включают fixed order0/1/2;
zero_division=0. Macro F1 включает все заявленные классы; balanced accuracy — mean
recall поддержанных классов (в принятом CV поддержаны все). Majority baseline
вычисляется по training counts; ties lowest index. Decision — argmax softmax,
при равенстве lowest index; это не calibrated confidence.

`recompute_fold_metrics(private_rows,mode="multiclass")` либо mode в private rows
восстанавливают fold metrics/baseline из3-class probability vectors; summary
пересчитывается тем же aggregate helper. Binary rows без mode остаются совместимыми.
Analyzer применяет typed allowlists также к per-class3-class aggregates и
macro-F1 epoch history; unknown private extras не переходят в stdout/JSON/CSV/plots.
Проверки: [multiclass research](../../../tests/test_multiclass_research.py) включают
настоящий CPU optimizer update, save/reload/aligned predictions, recomputation,
mode/checkpoint refusal, exactscore tie, CLI/notebook и private extra guards.
Этот developmentCV не является независимой test-оценкой или достижением quality goal.
