# 🚀 Quick Start - TMJ Detection System

## Что было сделано

✅ **Backend (Swift Vapor)**
- Endpoint для загрузки множественных DICOM файлов
- Интеграция с ML Service
- API для получения результатов детекции с координатами TMJ
- Сохранение volumeShape для визуализации

✅ **ML Service (Python FastAPI)**  
- Автоматическое определение типа модели (small/large) из config.json
- Поддержка Apple Silicon (MPS)
- Возврат размера volume вместе с результатами
- Улучшенная загрузка модели с логированием метрик

✅ **iOS App (SwiftUI)**
- Новый `TMJDetectionView` для загрузки папки DICOM
- Визуализация координат левого и правого TMJ
- Отображение bounding box и центра
- Красивый UI с chip-компонентами для координат

✅ **Обучение модели**
- Модель обучается на расширенном датасете (37 снимков)
- Текущие метрики: **97.34 px MAE** (Epoch 3)
- Улучшение: 156.89 → 106.67 → **97.34 px**

## Как запустить систему

### 1. ML Service (Terminal 1)
```bash
cd MLService
source venv/bin/activate
python app.py
```

Или используйте существующую модель:
```bash
export MODEL_PATH="experiments/detector_20251124_175805/best_model.pth"
python app.py
```

Проверка:
```bash
curl http://localhost:8001/health
```

### 2. Backend (Terminal 2)
```bash
cd Backend
swift run
```

Проверка:
```bash
curl http://localhost:8080/api/health
```

### 3. iOS App
1. Открыть `iOSApp/MasterDoctor/MasterDoctor.xcodeproj` в Xcode
2. Запустить на симуляторе или устройстве
3. Нажать "Select DICOM Folder"
4. Выбрать папку с `.dcm` файлами
5. Нажать "Start TMJ Detection"
6. Ждать результаты!

## Что увидите

### iOS App UI
```
┌─────────────────────────────────────┐
│  📁 Select DICOM Folder             │
│  → 22 DICOM files selected          │
│                                      │
│  🧠 Start TMJ Detection             │
│                                      │
│  📊 Detection Results                │
│  ┌─────────────────────────────┐   │
│  │ Left TMJ                    │   │
│  │ Center: Z:245.2 Y:123.4...  │   │
│  │ BBox: [220, 100, 350] →...  │   │
│  └─────────────────────────────┘   │
│                                      │
│  ┌─────────────────────────────┐   │
│  │ Right TMJ                   │   │
│  │ Center: Z:245.8 Y:645.1...  │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Мониторинг обучения

Модель сейчас обучается! Проверить прогресс:

```bash
# Последние метрики
tail -30 ~/.cursor/projects/Users-tzopiz-Developer-MasterProject/terminals/10.txt | grep "Epoch\|MAE"

# Или используйте скрипт (если создавали)
cd MLService
./venv/bin/python check_progress.py
```

## Структура ответа API

```json
{
  "taskId": "uuid",
  "status": "completed",
  "tmjLeft": {
    "center": [245.2, 123.4, 350.1],
    "bbox": [220, 100, 330, 270, 150, 370]
  },
  "tmjRight": {
    "center": [245.8, 645.1, 355.3],
    "bbox": [220, 620, 335, 270, 670, 375]
  },
  "volumeShape": [576, 768, 768]
}
```

## Файлы изменены

### Backend
- ✏️ `Models/AnalysisResult.swift` - добавлено поле `volumeShape`
- ✏️ `Controllers/AnalysisController.swift` - обновлен ответ с новыми полями
- ✏️ `Services/MLServiceClient.swift` - парсинг `volumeShape`

### ML Service  
- ✏️ `services/detector_service.py` - автоопределение типа модели, MPS support
- ✏️ `app.py` - возврат `volume_shape` в ответе

### iOS App
- ✏️ `Models/AnalysisModels.swift` - добавлены `BoundingBox`, `tmjLeft`, `tmjRight`
- ✏️ `Models/AnalyticsResponse.swift` - новые поля в ответе
- ✏️ `Models/Endpoint/AnalysisEndpoint.swift` - поддержка множественных файлов
- ✏️ `Views/AnalysisResultView.swift` - парсинг новых полей
- ✨ `Views/TMJDetectionView.swift` - **НОВЫЙ VIEW** с UI для загрузки папок
- ✏️ `MasterDoctorApp.swift` - использует `TMJDetectionView`

## Troubleshooting

**ML Service не находит модель:**
```bash
ls -lh experiments/detector_*/best_model.pth
export MODEL_PATH="experiments/detector_20251124_175805/best_model.pth"
```

**iOS не может выбрать папку:**
- Проверьте права в `Info.plist`
- Убедитесь что используете `.fileImporter` с `.folder`

**Backend не подключается к ML Service:**
```bash
# Проверить
curl http://localhost:8001/health

# Если нужно, задать URL
export ML_SERVICE_URL="http://localhost:8001"
```

## Next Steps

1. Дождаться окончания обучения (цель: MAE < 50px)
2. Добавить 3D визуализацию volume
3. Экспорт результатов
4. Batch processing для множества пациентов

---

**Текущий статус обучения:** Epoch 3/150, MAE = 97.34 px 📈

Система готова к работе! 🎉

