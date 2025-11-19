import Vapor
import Fluent

struct DICOMController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        let dicom = routes.grouped("api", "dicom")
        dicom.post("upload", use: upload)
    }
    
    func upload(req: Request) async throws -> UploadResponse {
        let file = try req.content.decode(FileUpload.self)
        
        // Validate file extension
        guard file.filename.hasSuffix(".dcm") || file.filename.hasSuffix(".DCM") else {
            throw Abort(.badRequest, reason: "Only DICOM files (.dcm) are accepted")
        }
        
        // Create uploads directory if it doesn't exist
        let uploadsDir = req.application.directory.workingDirectory + "uploads/"
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: uploadsDir) {
            try fileManager.createDirectory(atPath: uploadsDir, withIntermediateDirectories: true)
        }
        
        // Generate unique filename
        let taskID = UUID()
        let uniqueFilename = "\(taskID.uuidString)_\(file.filename)"
        let filePath = uploadsDir + uniqueFilename
        
        // Save file to disk
        try await req.fileio.writeFile(file.data, at: filePath)
        
        // Create task in database
        let task = AnalysisTask(
            id: taskID,
            dicomFilename: file.filename,
            dicomPath: filePath,
            status: .pending
        )
        try await task.save(on: req.db)
        
        // Trigger ML processing (async)
        Task {
            await processTask(taskID: taskID, filePath: filePath, app: req.application)
        }
        
        req.logger.info("DICOM file uploaded: \(file.filename), task ID: \(taskID)")
        
        return UploadResponse(
            taskId: taskID,
            status: "uploaded",
            message: "File uploaded successfully. Processing started."
        )
    }
    
    private func processTask(taskID: UUID, filePath: String, app: Application) async {
        do {
            // Update task status to processing
            guard let task = try await AnalysisTask.find(taskID, on: app.db) else {
                app.logger.error("Task not found: \(taskID)")
                return
            }
            
            task.status = .processing
            try await task.save(on: app.db)
            
            // Call ML Service
            let mlService = MLServiceClient(app: app)
            let result = try await mlService.processFile(taskID: taskID, filePath: filePath)
            
            // Save results
            let analysisResult = AnalysisResult(
                taskID: taskID,
                slicesData: result.slicesData,
                masksData: result.masksData,
                parameters: result.parameters,
                diagnosis: result.diagnosis
            )
            try await analysisResult.save(on: app.db)
            
            // Update task status to completed
            task.status = .completed
            try await task.save(on: app.db)
            
            app.logger.info("Task completed: \(taskID)")
        } catch {
            app.logger.error("Task failed: \(taskID), error: \(error)")
            
            // Update task status to failed
            if let task = try? await AnalysisTask.find(taskID, on: app.db) {
                task.status = .failed
                task.errorMessage = error.localizedDescription
                try? await task.save(on: app.db)
            }
        }
    }
}

struct FileUpload: Content {
    var filename: String
    var data: ByteBuffer
}

struct UploadResponse: Content {
    let taskId: UUID
    let status: String
    let message: String
}

