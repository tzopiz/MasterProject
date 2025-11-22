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

## 2. Загрузка DICOM файла (Start Analysis)

Используется `multipart/form-data`.

### Через curl (с локальным файлом)

```bash
curl -X POST http://localhost:8080/api/analysis \
  -F "file=@/path/to/your/file.dcm"
```

Ответ:
```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Через Swift (URLSession)

```swift
import Foundation

struct UploadResponse: Codable {
    let taskId: UUID
}

func uploadDICOM(fileURL: URL) async throws -> UUID {
    let url = URL(string: "http://localhost:8080/api/analysis")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    
    var data = Data()
    let filename = fileURL.lastPathComponent
    let fileData = try Data(contentsOf: fileURL)
    
    data.append("--\(boundary)\r\n".data(using: .utf8)!)
    data.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
    data.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
    data.append(fileData)
    data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
    
    let (responseData, _) = try await URLSession.shared.upload(for: request, from: data)
    let response = try JSONDecoder().decode(UploadResponse.self, from: responseData)
    
    return response.taskId
}
```

## 3. Проверка статуса задачи (Polling)

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
*Примечание: поля `slicesData`, `masksData`, `parameters`, `diagnosis` приходят как JSON-строки, которые нужно распарсить.*

## 5. Пример полного цикла (Polling)

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
