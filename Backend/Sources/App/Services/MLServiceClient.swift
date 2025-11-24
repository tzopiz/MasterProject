import Vapor
import AsyncHTTPClient
import Foundation
import NIO
import NIOFoundationCompat

struct MLServiceClient {
    let app: Application
    
    // ML Service URL (can be configured via environment variable)
    private var mlServiceURL: String {
        Environment.get("ML_SERVICE_URL") ?? "http://localhost:8001"
    }
    
    func processSeries(taskID: UUID, directoryPath: String) async throws -> MLProcessingResult {
        let client = HTTPClient.shared
        let boundary = "Boundary-\(UUID().uuidString)"
        
        // Get all files in directory
        let fileManager = FileManager.default
        let fileURLs = try fileManager.contentsOfDirectory(at: URL(fileURLWithPath: directoryPath), 
                                                          includingPropertiesForKeys: nil)
                                      .filter { $0.pathExtension.lowercased() == "dcm" }
        
        guard !fileURLs.isEmpty else {
            throw Abort(.badRequest, reason: "No DICOM files found in directory")
        }
        
        // Construct Multipart Body
        var body = ByteBufferAllocator().buffer(capacity: 0)
        
        // 1. Add Task ID field
        body.writeString("--\(boundary)\r\n")
        body.writeString("Content-Disposition: form-data; name=\"task_id\"\r\n\r\n")
        body.writeString("\(taskID.uuidString)\r\n")
        
        // 2. Add Files
        for fileURL in fileURLs {
            let filename = fileURL.lastPathComponent
            guard let fileData = try? Data(contentsOf: fileURL) else { continue }
            
            body.writeString("--\(boundary)\r\n")
            body.writeString("Content-Disposition: form-data; name=\"files\"; filename=\"\(filename)\"\r\n")
            body.writeString("Content-Type: application/dicom\r\n\r\n")
            body.writeBytes(fileData)
            body.writeString("\r\n")
        }
        
        body.writeString("--\(boundary)--\r\n")
        
        // Send Request
        var request = HTTPClientRequest(url: "\(mlServiceURL)/process")
        request.method = .POST
        request.headers.add(name: "Content-Type", value: "multipart/form-data; boundary=\(boundary)")
        request.body = .bytes(body)
        
        app.logger.info("Sending \(fileURLs.count) files to ML Service for task: \(taskID)")
        
        // Increase timeout for large uploads
        let response = try await client.execute(request, timeout: .minutes(10))
        
        guard response.status == .ok else {
            let bodyBytes = try await response.body.collect(upTo: 1024 * 1024) // 1MB max
            let errorMessage = String(buffer: bodyBytes)
            app.logger.error("ML Service error: \(errorMessage)")
            throw Abort(.internalServerError, reason: "ML Service error: \(errorMessage)")
        }
        
        let responseBody = try await response.body.collect(upTo: 50 * 1024 * 1024) // 50MB max
        let decoder = JSONDecoder()
        
        let mlResponse = try decoder.decode(MLServiceResponse.self, from: responseBody)
        
        return MLProcessingResult(from: mlResponse)
    }
}


// Response structures from ML Service (Updated for 3D)
struct MLServiceResponse: Codable {
    let taskId: String
    let status: String
    let tmj: TMJResult?
    let errorMessage: String?
    
    enum CodingKeys: String, CodingKey {
        case taskId = "task_id"
        case status
        case tmj
        case errorMessage = "error_message"
    }
}

struct TMJResult: Codable {
    let left: BoundingBox
    let right: BoundingBox
}

struct BoundingBox: Codable {
    let center: [Float]
    let bbox: [Int]
}


// Internal result format (Simplified for now)
struct MLProcessingResult {
    let leftTMJ: BoundingBox?
    let rightTMJ: BoundingBox?
    let errorMessage: String?
    
    init(from response: MLServiceResponse) {
        self.leftTMJ = response.tmj?.left
        self.rightTMJ = response.tmj?.right
        self.errorMessage = response.errorMessage
    }
}

// Extend HTTPClient for shared instance
extension HTTPClient {
    static let shared = HTTPClient(eventLoopGroupProvider: .singleton)
}
