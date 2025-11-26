import Vapor
import Foundation
import NIOCore

struct DICOMStorageService {
    let app: Application
    
    func saveFile(data: ByteBuffer, filename: String) async throws -> String {
        let uploadsDir = app.directory.workingDirectory + "uploads/"
        let fileManager = FileManager.default
        
        // Create uploads directory if it doesn't exist
        if !fileManager.fileExists(atPath: uploadsDir) {
            try fileManager.createDirectory(atPath: uploadsDir, withIntermediateDirectories: true)
        }
        
        // Generate unique filename
        let uniqueFilename = "\(UUID().uuidString)_\(filename)"
        let filePath = uploadsDir + uniqueFilename
        
        // Convert ByteBuffer to Data
        guard let fileData = data.getData(at: 0, length: data.readableBytes) else {
             throw Abort(.internalServerError, reason: "Failed to read file data")
        }
        
        // Write file synchronously (for now, assuming reasonable file sizes or offload to thread pool manually if needed)
        // To be fully non-blocking, we should use app.threadPool.runIfActive
        
        try await app.threadPool.runIfActive(eventLoop: app.eventLoopGroup.next()) {
            try fileData.write(to: URL(fileURLWithPath: filePath))
        }.get()
        
        return filePath
    }
    
    func saveSeries(files: [File], taskID: UUID) async throws -> String {
        let uploadsDir = app.directory.workingDirectory + "uploads/"
        let taskDir = uploadsDir + taskID.uuidString + "/"
        let fileManager = FileManager.default
        
        // Create task directory
        if !fileManager.fileExists(atPath: taskDir) {
            try fileManager.createDirectory(atPath: taskDir, withIntermediateDirectories: true)
        }
        
        // Write all files
        // For a large series, we might want to parallelize or offload this
        try await app.threadPool.runIfActive(eventLoop: app.eventLoopGroup.next()) {
            for file in files {
                let filePath = taskDir + file.filename
                if let data = file.data.getData(at: 0, length: file.data.readableBytes) {
                    try? data.write(to: URL(fileURLWithPath: filePath))
                }
            }
        }.get()
        
        return taskDir
    }
    
    func deleteSeries(path: String) throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: path) {
            try fileManager.removeItem(atPath: path)
        }
    }
    
    func deleteFile(path: String) throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: path) {
            try fileManager.removeItem(atPath: path)
        }
    }
}
