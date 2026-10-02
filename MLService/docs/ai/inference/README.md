# HTTP: текущая реализация

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
