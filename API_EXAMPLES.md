# Примеры использования API

## Предварительные требования

Убедитесь, что оба сервиса запущены:
- Backend на `http://localhost:8080`
- ML Service на `http://localhost:8001`

## 1. Health Checks

### Проверка Backend
```bash
curl http://localhost:8080/health
```

Ответ:
```json
{
  "status": "ok",
  "service": "vapor-backend",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Проверка ML Service
```bash
curl http://localhost:8001/health
```

Ответ:
```json
{
  "status": "ok",
  "service": "ml-service",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_loaded": true
}
```

### Проверка статуса модели
```bash
curl http://localhost:8001/models/status
```

Ответ:
```json
{
  "model_loaded": true,
  "model_type": "segmentation",
  "model_path": "models/segmentation_model.pth"
}
```

## 2. Загрузка DICOM файла

### Через curl (с локальным файлом)

```bash
curl -X POST http://localhost:8080/api/dicom/upload \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "patient_scan.dcm",
    "data": "'"$(base64 -i path/to/your/file.dcm)"'"
  }'
```

Ответ:
```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploaded",
  "message": "File uploaded successfully. Processing started."
}
```

### Через Python

```python
import requests
import base64

# Загрузить DICOM файл
with open('path/to/file.dcm', 'rb') as f:
    dicom_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    'http://localhost:8080/api/dicom/upload',
    json={
        'filename': 'patient_scan.dcm',
        'data': dicom_data
    }
)

task_id = response.json()['taskId']
print(f"Task ID: {task_id}")
```

### Через Swift

```swift
import Foundation

struct FileUpload: Codable {
    let filename: String
    let data: String  // base64
}

struct UploadResponse: Codable {
    let taskId: UUID
    let status: String
    let message: String
}

func uploadDICOM(fileURL: URL) async throws -> UUID {
    let data = try Data(contentsOf: fileURL)
    let base64 = data.base64EncodedString()
    
    let upload = FileUpload(
        filename: fileURL.lastPathComponent,
        data: base64
    )
    
    let url = URL(string: "http://localhost:8080/api/dicom/upload")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONEncoder().encode(upload)
    
    let (data, _) = try await URLSession.shared.data(for: request)
    let response = try JSONDecoder().decode(UploadResponse.self, from: data)
    
    return response.taskId
}
```

## 3. Проверка статуса задачи

```bash
curl http://localhost:8080/api/analysis/{taskId}/status
```

Возможные статусы:
- `pending` - в очереди
- `processing` - обрабатывается
- `completed` - завершено
- `failed` - ошибка

Ответ:
```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "errorMessage": null
}
```

## 4. Получение результатов анализа

```bash
curl http://localhost:8080/api/analysis/{taskId}
```

Ответ (когда status = "completed"):
```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "errorMessage": null,
  "slicesData": "{\"orthogonal\":[\"base64_image1\",\"base64_image2\"],\"sagittal\":[...],\"frontal\":[...]}",
  "masksData": "{\"orthogonal\":[\"base64_mask1\",\"base64_mask2\"],\"sagittal\":[...],\"frontal\":[...]}",
  "parameters": "{\"fossa_height\":12.5,\"head_height\":8.3,\"width\":15.2,\"additional_params\":{...}}",
  "diagnosis": "{\"status\":\"normal\",\"confidence\":0.85,\"recommendations\":[...],\"disclaimer\":\"...\"}"
}
```

## 5. Polling для получения результатов

### Python пример

```python
import requests
import time
import json

def wait_for_results(task_id, timeout=300, interval=2):
    """
    Ожидание завершения обработки
    
    Args:
        task_id: UUID задачи
        timeout: максимальное время ожидания (секунды)
        interval: интервал между проверками (секунды)
    
    Returns:
        dict: результаты анализа
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Проверить статус
        response = requests.get(
            f'http://localhost:8080/api/analysis/{task_id}/status'
        )
        
        if response.status_code != 200:
            raise Exception(f"Error checking status: {response.text}")
        
        status_data = response.json()
        status = status_data['status']
        
        print(f"Status: {status}")
        
        if status == 'completed':
            # Получить результаты
            response = requests.get(
                f'http://localhost:8080/api/analysis/{task_id}'
            )
            return response.json()
        
        elif status == 'failed':
            raise Exception(f"Task failed: {status_data.get('errorMessage')}")
        
        # Подождать перед следующей проверкой
        time.sleep(interval)
    
    raise TimeoutError(f"Task did not complete within {timeout} seconds")

# Использование
try:
    results = wait_for_results('550e8400-e29b-41d4-a716-446655440000')
    
    # Парсить результаты
    parameters = json.loads(results['parameters'])
    diagnosis = json.loads(results['diagnosis'])
    
    print(f"Fossa height: {parameters['fossa_height']} mm")
    print(f"Head height: {parameters['head_height']} mm")
    print(f"Width: {parameters['width']} mm")
    print(f"Status: {diagnosis['status']}")
    print(f"Confidence: {diagnosis['confidence']}")
    print("\nRecommendations:")
    for rec in diagnosis['recommendations']:
        print(f"  - {rec}")
    
except Exception as e:
    print(f"Error: {e}")
```

### Swift пример

```swift
func pollForResults(taskId: UUID) async throws -> AnalysisResponse {
    let maxAttempts = 150  // 5 минут при интервале 2 секунды
    let interval: UInt64 = 2_000_000_000  // 2 секунды в наносекундах
    
    for _ in 0..<maxAttempts {
        let url = URL(string: "http://localhost:8080/api/analysis/\(taskId.uuidString)/status")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let status = try JSONDecoder().decode(StatusResponse.self, from: data)
        
        switch status.status {
        case "completed":
            // Получить полные результаты
            let resultsURL = URL(string: "http://localhost:8080/api/analysis/\(taskId.uuidString)")!
            let (resultsData, _) = try await URLSession.shared.data(from: resultsURL)
            return try JSONDecoder().decode(AnalysisResponse.self, from: resultsData)
            
        case "failed":
            throw NSError(domain: "Analysis", code: -1, 
                         userInfo: [NSLocalizedDescriptionKey: status.errorMessage ?? "Unknown error"])
            
        default:
            // pending или processing - продолжить ожидание
            try await Task.sleep(nanoseconds: interval)
        }
    }
    
    throw NSError(domain: "Analysis", code: -2, 
                 userInfo: [NSLocalizedDescriptionKey: "Timeout waiting for results"])
}
```

## 6. Полный workflow

```python
import requests
import base64
import json
import time

# 1. Загрузить DICOM файл
with open('patient_scan.dcm', 'rb') as f:
    dicom_data = base64.b64encode(f.read()).decode('utf-8')

upload_response = requests.post(
    'http://localhost:8080/api/dicom/upload',
    json={
        'filename': 'patient_scan.dcm',
        'data': dicom_data
    }
)

task_id = upload_response.json()['taskId']
print(f"Task created: {task_id}")

# 2. Ожидание обработки
print("Waiting for processing...")
while True:
    status_response = requests.get(
        f'http://localhost:8080/api/analysis/{task_id}/status'
    )
    status = status_response.json()['status']
    print(f"Status: {status}")
    
    if status == 'completed':
        break
    elif status == 'failed':
        print(f"Error: {status_response.json().get('errorMessage')}")
        exit(1)
    
    time.sleep(2)

# 3. Получить результаты
results_response = requests.get(
    f'http://localhost:8080/api/analysis/{task_id}'
)
results = results_response.json()

# 4. Обработать результаты
parameters = json.loads(results['parameters'])
diagnosis = json.loads(results['diagnosis'])

print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
print(f"\nГеометрические параметры:")
print(f"  Высота суставной ямки: {parameters['fossa_height']:.2f} мм")
print(f"  Высота суставной головки: {parameters['head_height']:.2f} мм")
print(f"  Ширина сустава: {parameters['width']:.2f} мм")

print(f"\nДиагноз: {diagnosis['status']}")
print(f"Уверенность: {diagnosis['confidence']:.2%}")

print("\nРекомендации:")
for rec in diagnosis['recommendations']:
    print(f"  • {rec}")

print(f"\n{diagnosis['disclaimer']}")
```

## 7. Декодирование изображений из base64

```python
import base64
from PIL import Image
from io import BytesIO

def decode_base64_image(base64_string):
    """Декодировать base64 в PIL Image"""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

# Использование
slices_data = json.loads(results['slicesData'])
masks_data = json.loads(results['masksData'])

# Показать первый ортогональный срез
if slices_data['orthogonal']:
    slice_image = decode_base64_image(slices_data['orthogonal'][0])
    slice_image.show()
    
# Показать первую маску
if masks_data['orthogonal']:
    mask_image = decode_base64_image(masks_data['orthogonal'][0])
    mask_image.show()
```

## Troubleshooting

### Backend не запускается

```bash
cd Backend
swift package clean
swift build
```

### ML Service не может найти модель

```bash
# Проверить наличие модели
ls -la MLService/models/

# Если модели нет, сервис работает в dummy mode
# Можно запустить без модели для тестирования
```

### Ошибка при загрузке большого файла

Увеличить лимит размера в `Backend/Sources/App/configure.swift`:
```swift
app.routes.defaultMaxBodySize = "1gb"  // Увеличить до 1GB
```

### Таймаут при обработке

Увеличить таймаут в `Backend/Sources/App/Services/MLServiceClient.swift`:
```swift
let response = try await client.execute(request, timeout: .minutes(30))
```

