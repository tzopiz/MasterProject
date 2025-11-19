import Vapor
import AsyncHTTPClient
import Foundation

struct MLServiceClient {
    let app: Application
    
    // ML Service URL (can be configured via environment variable)
    private var mlServiceURL: String {
        Environment.get("ML_SERVICE_URL") ?? "http://localhost:8001"
    }
    
    func processFile(taskID: UUID, filePath: String) async throws -> MLProcessingResult {
        let client = HTTPClient.shared
        
        // Prepare request payload
        let payload = MLProcessingRequest(
            dicomPath: filePath,
            taskId: taskID.uuidString
        )
        
        let encoder = JSONEncoder()
        let requestBody = try encoder.encode(payload)
        
        var request = HTTPClientRequest(url: "\(mlServiceURL)/process")
        request.method = .POST
        request.headers.add(name: "Content-Type", value: "application/json")
        request.body = .bytes(requestBody)
        
        app.logger.info("Sending request to ML Service for task: \(taskID)")
        
        let response = try await client.execute(request, timeout: .minutes(10))
        
        guard response.status == .ok else {
            let bodyBytes = try await response.body.collect(upTo: 1024 * 1024) // 1MB max
            let errorMessage = String(buffer: bodyBytes)
            app.logger.error("ML Service error: \(errorMessage)")
            throw Abort(.internalServerError, reason: "ML Service error: \(errorMessage)")
        }
        
        let responseBody = try await response.body.collect(upTo: 50 * 1024 * 1024) // 50MB max
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        
        let mlResponse = try decoder.decode(MLServiceResponse.self, from: responseBody)
        
        // Convert to internal result format
        return try MLProcessingResult(from: mlResponse)
    }
}

// Request structures
struct MLProcessingRequest: Codable {
    let dicomPath: String
    let taskId: String
    
    enum CodingKeys: String, CodingKey {
        case dicomPath = "dicom_path"
        case taskId = "task_id"
    }
}

// Response structures from ML Service
struct MLServiceResponse: Codable {
    let taskId: String
    let status: String
    let slices: SlicesData?
    let masks: MasksData?
    let parameters: GeometricParameters?
    let diagnosis: DiagnosisData?
    
    enum CodingKeys: String, CodingKey {
        case taskId = "task_id"
        case status
        case slices
        case masks
        case parameters
        case diagnosis
    }
}

struct SlicesData: Codable {
    let orthogonal: [String]?
    let sagittal: [String]?
    let frontal: [String]?
}

struct MasksData: Codable {
    let orthogonal: [String]?
    let sagittal: [String]?
    let frontal: [String]?
}

struct GeometricParameters: Codable {
    let fossaHeight: Double?
    let headHeight: Double?
    let width: Double?
    let additionalParams: [String: Double]?
    
    enum CodingKeys: String, CodingKey {
        case fossaHeight = "fossa_height"
        case headHeight = "head_height"
        case width
        case additionalParams = "additional_params"
    }
}

struct DiagnosisData: Codable {
    let status: String
    let confidence: Double?
    let recommendations: [String]?
    let disclaimer: String?
}

// Internal result format
struct MLProcessingResult {
    let slicesData: String?
    let masksData: String?
    let parameters: String?
    let diagnosis: String?
    
    init(from response: MLServiceResponse) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        if let slices = response.slices {
            self.slicesData = String(data: try encoder.encode(slices), encoding: .utf8)
        } else {
            self.slicesData = nil
        }
        
        if let masks = response.masks {
            self.masksData = String(data: try encoder.encode(masks), encoding: .utf8)
        } else {
            self.masksData = nil
        }
        
        if let params = response.parameters {
            self.parameters = String(data: try encoder.encode(params), encoding: .utf8)
        } else {
            self.parameters = nil
        }
        
        if let diag = response.diagnosis {
            self.diagnosis = String(data: try encoder.encode(diag), encoding: .utf8)
        } else {
            self.diagnosis = nil
        }
    }
}

// Extend HTTPClient for shared instance
extension HTTPClient {
    static let shared = HTTPClient(eventLoopGroupProvider: .singleton)
}

