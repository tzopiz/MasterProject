# TMJ Analysis ML Service

Сервис для автоматического анализа КЛКТ (CBCT) снимков ВНЧС (TMJ).
Этот проект предоставляет API для анализа DICOM файлов, а также инструменты для обучения моделей детекции и сегментации.

---

## 🚀 Текущий пайплайн (Pipeline)

Весь процесс разделен на 3 основных этапа:

### 1. Подготовка данных (Dataset)
Организация сырых DICOM файлов в структурированный вид.
- **Инструмент:** `tools/organize_dataset.py`
- **Вход:** Папка с сырыми DICOM файлами.
- **Выход:** Структура `data/dataset/study_XXXX/`.

```bash
python tools/organize_dataset.py --input <raw_data_folder> --output data/dataset
```

### 2. Детекция (Localization)
Нахождение координат центра сустава (ROI - Region of Interest) на полном 3D снимке.
- **Разметка:** `tools/roi_annotation_tool.py` (GUI для клика по центрам суставов).
- **Обучение:** `train_detector.py` (Обучает модель находить координаты).
- **Инференс:** `tools/auto_crop_from_detector.py` (Использует обученную модель для вырезания кропов 128x128x128).

**Результат:** JSON файлы с координатами и вырезанные 3D кубы суставов в `data/auto_crops/`.

### 3. Сегментация (Segmentation)
Точное выделение костных структур внутри вырезанного кропа.
- **Разметка:** Ручная разметка кропов в **ITK-SNAP**.
- **Обучение:** `train_3d.py` (Обучает 3D U-Net на кропах).
- **Модель:** `models/unet_3d.py`.

---

## 🛠 Установка и Запуск

### Требования
- Python 3.9+
- macOS / Linux (рекомендуется)
- CUDA (опционально, для ускорения обучения)

### Быстрый старт
```bash
cd MLService
./start.sh
```

Или вручную:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Сервис будет доступен по адресу: `http://localhost:8001`

---

## 🧠 Обучение моделей

### TMJ Detector (Этап 2)
Обучение модели для поиска суставов на полном снимке.

```bash
# 1. Разметка данных (создает JSON файлы)
python tools/roi_annotation_tool.py

# 2. Обучение модели
python train_detector.py --epochs 200 --batch_size 2

# 3. Мониторинг (в другом терминале)
tail -f experiments/detector_*/training.log
```

### 3D Segmentation (Этап 3)
Обучение модели сегментации на подготовленных кропах.

```bash
# 1. Подготовка кропов (используя обученный детектор)
python tools/auto_crop_from_detector.py --model experiments/detector_LATEST/best_model.pth

# 2. Обучение 3D U-Net
python train_3d.py --data_dir data/auto_crops --epochs 100
```

---

## 📡 API Reference

Сервис предоставляет REST API для интеграции с Backend.

### 1. Health Check
`GET /health`
Проверка работоспособности сервиса и статуса загрузки модели.

### 2. Start Analysis
`POST /process`
Запуск анализа DICOM файла.
**Body:**
```json
{
  "dicom_path": "/path/to/file.dcm",
  "task_id": "uuid-string"
}
```

### 3. Check Model Status
`GET /models/status`
Информация о загруженной модели (тип, путь).

---

## 🛠 Инструменты (Tools)

В папке `tools/` находятся утилиты для работы с данными:

- `roi_annotation_tool.py`: GUI приложение для быстрой разметки центров суставов.
- `organize_dataset.py`: Скрипт для сортировки и переименования DICOM файлов.
- `auto_crop_from_detector.py`: Автоматическое создание кропов с использованием детектора.
- `manual_crop_tool.py`: (Legacy) Ручное создание кропов.
- `create_portable_tool.py`: Создание portable-версии разметчика для передачи врачам/коллегам.
- `visualize_detector.py`: Визуальная проверка работы детектора (рисует bounding box на снимке).
- `extract_crops_from_annotations.py`: Создание кропов на основе JSON аннотаций (без детектора).

---

## 📊 Мониторинг обучения

Для слежения за процессом обучения (Detector или Segmentation) используйте логи в терминале или файлы в `experiments/`.

**Полезные команды:**
- `tail -f <logfile>`: Следить в реальном времени.
- `grep "best" <logfile>`: Найти лучшие эпохи.
- `grep "Val MAE" <logfile>`: Посмотреть метрики валидации.

Подробнее см. `TRAINING_MONITORING.md` (если доступен).

---

## 📦 Deployment (Развертывание)

### Docker
Создайте `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Переменные окружения (Production)
- `MODEL_PATH`: Путь к файлу модели (.pth).
- `API_PORT`: Порт сервиса (default: 8001).
- `WORKERS`: Количество воркеров uvicorn.

---

## 📚 Legacy & Manual Workflows

### Ручная сегментация (ITK-SNAP)
Для создания Ground Truth масок:
1. Откройте кроп (`study_XXXX_left.nii.gz`) в ITK-SNAP.
2. Создайте сегментацию (Label 1).
3. Сохраните как `study_XXXX_left_seg.nii.gz`.

### Анализ серии (Legacy Script)
Ранее использовался скрипт `analyze_dicom_series.py` для генерации превью (axial/coronal/sagittal) из папки с DICOM. Сейчас эта логика частично интегрирована в основной пайплайн или заменена инструментами визуализации.
