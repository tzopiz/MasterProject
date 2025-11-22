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
        
        // Save file using FileIO
        let fileHandle = try await app.fileio.openFile(
            path: filePath,
            mode: .write,
            flags: .allowFileCreation(),
            eventLoop: app.eventLoopGroup.next()
        ).get()
        
        try await app.fileio.write(
            fileHandle: fileHandle,
            buffer: data,
            eventLoop: app.eventLoopGroup.next()
        ).get()
        
        try fileHandle.close()
        
        return filePath
    }
    
    func deleteFile(path: String) throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: path) {
            try fileManager.removeItem(atPath: path)
        }
    }
}

