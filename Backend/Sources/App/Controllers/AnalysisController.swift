import Vapor
import Fluent

struct AnalysisController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        let analysis = routes.grouped("api", "analysis")
        
        // Upload files (series) and start analysis
        analysis.on(.POST, body: .collect(maxSize: "100mb"), use: uploadSeries)
        
        // Get results
        analysis.get(":taskId", use: getResult)
    }
    
    // MARK: - Handlers
    
    func uploadSeries(req: Request) async throws -> AnalysisUploadResponse {
        // 1. Decode list of files
        // Expecting multipart/form-data with multiple 'files' fields
        
        // Manually iterate over multipart parts if needed, or use Vapor's decoder
        // Vapor handles [File] if named "files[]" or similar, but simpler is usually just iterate content
        
        // Let's try standard decoding if client sends 'files' array
        struct SeriesUploadRequest: Content {
            var files: [File]
        }
        
        let input = try req.content.decode(SeriesUploadRequest.self)
        
        guard !input.files.isEmpty else {
            throw Abort(.badRequest, reason: "No files provided")
        }
        
        // 2. Create Task
        let taskId = UUID()
        
        // 3. Save files to disk (Folder: uploads/{taskId}/)
        let storage = DICOMStorageService(app: req.application)
        let savedPath = try await storage.saveSeries(files: input.files, taskID: taskId)
        
        // 4. Save Task to DB
        let task = AnalysisTask(
            id: taskId,
            dicomFilename: "series_\(input.files.count)_files", // Placeholder name
            dicomPath: savedPath,
            status: .pending
        )
        try await task.save(on: req.db)
        
        // 5. Trigger Background Processing
        let app = req.application
        
        Task.detached {
            do {
                try await self.processTask(app: app, taskId: taskId, directoryPath: savedPath)
            } catch {
                app.logger.error("Failed to process task \(taskId): \(error)")
                try? await self.updateStatus(app: app, taskId: taskId, status: .failed, error: String(describing: error))
            }
        }
        
        return AnalysisUploadResponse(taskId: taskId)
    }
    
    func getResult(req: Request) async throws -> AnalysisResponse {
        guard let taskIdString = req.parameters.get("taskId"),
              let taskId = UUID(uuidString: taskIdString) else {
            throw Abort(.badRequest, reason: "Invalid task ID")
        }
        
        guard let task = try await AnalysisTask.find(taskId, on: req.db) else {
            throw Abort(.notFound, reason: "Task not found")
        }
        
        // Get results (optional)
        let result = try await AnalysisResult.query(on: req.db)
            .filter(\.$task.$id == taskId)
            .first()
        
        // Format dates
        let dateFormatter = ISO8601DateFormatter()
        
        return AnalysisResponse(
            taskId: taskId,
            status: task.status.rawValue,
            errorMessage: task.errorMessage,
            tmjLeft: result?.tmjLeft,
            tmjRight: result?.tmjRight,
            volumeShape: result?.volumeShape,
            createdAt: task.createdAt.map { dateFormatter.string(from: $0) },
            updatedAt: task.updatedAt.map { dateFormatter.string(from: $0) }
        )
    }
    
    
    // MARK: - Private Helpers
    
    private func processTask(app: Application, taskId: UUID, directoryPath: String) async throws {
        app.logger.info("Starting background processing for task: \(taskId)")
        
        // 1. Update status to processing
        try await updateStatus(app: app, taskId: taskId, status: .processing)
        
        // 2. Call ML Service (Multipart upload of all files)
        let mlClient = MLServiceClient(app: app)
        let mlResult = try await mlClient.processSeries(taskID: taskId, directoryPath: directoryPath)
        
        if let error = mlResult.errorMessage {
             try await updateStatus(app: app, taskId: taskId, status: .failed, error: error)
             return
        }
        
        // 3. Save Results
        // Convert BoundingBox to JSON string for storage
        let encoder = JSONEncoder()
        
        let leftStr = mlResult.leftTMJ.flatMap { try? String(data: encoder.encode($0), encoding: .utf8) }
        let rightStr = mlResult.rightTMJ.flatMap { try? String(data: encoder.encode($0), encoding: .utf8) }
        
        let result = AnalysisResult(
            taskID: taskId,
            tmjLeft: leftStr,
            tmjRight: rightStr,
            volumeShape: mlResult.volumeShape
        )
        try await result.save(on: app.db)
        
        // 4. Update status to completed
        try await updateStatus(app: app, taskId: taskId, status: .completed)
        
        app.logger.info("Task \(taskId) completed successfully")
    }
    
    private func updateStatus(app: Application, taskId: UUID, status: TaskStatus, error: String? = nil) async throws {
        guard let task = try await AnalysisTask.find(taskId, on: app.db) else { return }
        task.status = status
        if let error = error { task.errorMessage = error }
        task.updatedAt = Date()
        try await task.save(on: app.db)
    }
}

// DTOs

struct AnalysisUploadResponse: Content {
    var taskId: UUID
}

struct AnalysisResponse: Content {
    let taskId: UUID
    let status: String
    let errorMessage: String?
    let tmjLeft: String? // JSON String
    let tmjRight: String? // JSON String
    let volumeShape: [Int]? // [depth, height, width] for client visualization
    let createdAt: String?
    let updatedAt: String?
}
