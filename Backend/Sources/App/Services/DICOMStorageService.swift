import Vapor
import Foundation

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
        
        // Save file
        let fileIO = app.fileio
        try await fileIO.writeFile(data, at: filePath)
        
        return filePath
    }
    
    func deleteFile(path: String) throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: path) {
            try fileManager.removeItem(atPath: path)
        }
    }
}

