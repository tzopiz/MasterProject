# Оценка: что означает текущий отчёт

[Сагиттальная CV](../../../training/sagittal_binary_cv.py) применяет
StratifiedGroupKFold по patient_name. Страта пациента — max sagittal binary label
всех его сторон/серий; асимметричный пациент относится к положительной страте.
Loss обучает sagittal head; best epoch, scheduler и early stopping используют
val AUC того же фолда. Отдельного test в этом модуле нет.

После выбора state Youden threshold рассчитывается на train predictions тем же
augmented train loader. model.eval() отключает dropout/BN training, но не dataset
augmentation. Старый [holdout trainer](../../../train_binary_position_classifier.py)
выбирает mean val accuracy и калибрует thresholds на этой же val.

[Binary metrics](../../../training/utils/binary_metrics.py) применяют score>=threshold,
confusion order [0,1], f1_minority=F1 positive=1 независимо от фактического меньшинства.
Single-class AUC=NaN и Youden=0.5. CV summary пропускает NaN, считает mean/population
std; это не доверительный интервал. JSON заменяется после fold, nonfinite→null,
complete означает completed_folds>=n_splits, а не достижение целевого качества.
JSON/epoch JSONL не дают resume и не сохраняют лучший checkpoint/fold membership.

[Multiclass trainer](../../../train_tmj_position_classifier.py) — whole-volume
четырёхголовый baseline, сумма четырёх CE; checkpoint по mean val accuracy.
[2D trainer](../../../train.py) делит volumes, [3D](../../../train_3d.py) — crops через
random_split; на малом наборе 3D допускает train=val.
Существующие [fold tests](../../../tests/test_stratified_group_kfold.py) и
[metrics tests](../../../tests/test_binary_metrics.py) не проверяют генерализацию.
