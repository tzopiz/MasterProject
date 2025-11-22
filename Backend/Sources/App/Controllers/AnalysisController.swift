import Vapor
import Fluent

struct AnalysisController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        let analysis = routes.grouped("api", "analysis")
        
        // Upload file and start analysis
        analysis.on(.POST, body: .collect(maxSize: "50mb"), use: upload)
        
        // Get results
        analysis.get(":taskId", use: getResult)
        analysis.get(":taskId", "status", use: getStatus)
    }
    
    // MARK: - Handlers
    
    func upload(req: Request) async throws -> AnalysisUploadResponse {
        // 1. Decode file from multipart form
        let input = try req.content.decode(UploadRequest.self)
        let file = input.file
        
        guard file.data.readableBytes > 0 else {
            throw Abort(.badRequest, reason: "File is empty")
        }
        
        // 2. Save file to disk
        let storage = DICOMStorageService(app: req.application)
        let savedPath = try await storage.saveFile(data: file.data, filename: file.filename)
        
        // 3. Create Task in DB
        let task = AnalysisTask(
            dicomFilename: file.filename,
            dicomPath: savedPath,
            status: .pending
        )
        try await task.save(on: req.db)
        
        // 4. Trigger Processing in Background
        // We assume 'task.id' is populated after save
        guard let taskId = task.id else {
            throw Abort(.internalServerError, reason: "Failed to create task ID")
        }
        
        let app = req.application
        
        Task.detached {
            do {
                try await self.processTask(app: app, taskId: taskId, filePath: savedPath)
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
        
        // If task is not completed, return status only
        if task.status != .completed {
            return AnalysisResponse(
                taskId: taskId,
                status: task.status.rawValue,
                errorMessage: task.errorMessage
            )
        }
        
        // Get results
        let result = try await AnalysisResult.query(on: req.db)
            .filter(\.$task.$id == taskId)
            .first()
        
        return AnalysisResponse(
            taskId: taskId,
            status: task.status.rawValue,
            errorMessage: task.errorMessage,
            slicesData: result?.slicesData,
            masksData: result?.masksData,
            parameters: result?.parameters,
            diagnosis: result?.diagnosis
        )
    }
    
    func getStatus(req: Request) async throws -> StatusResponse {
        guard let taskIdString = req.parameters.get("taskId"),
              let taskId = UUID(uuidString: taskIdString) else {
            throw Abort(.badRequest, reason: "Invalid task ID")
        }
        
        guard let task = try await AnalysisTask.find(taskId, on: req.db) else {
            throw Abort(.notFound, reason: "Task not found")
        }
        
        return StatusResponse(
            taskId: taskId,
            status: task.status.rawValue,
            errorMessage: task.errorMessage
        )
    }
    
    // MARK: - Private Helpers
    
    private func processTask(app: Application, taskId: UUID, filePath: String) async throws {
        app.logger.info("Starting background processing for task: \(taskId)")
        
        // 1. Update status to processing
        try await updateStatus(app: app, taskId: taskId, status: .processing)
        
        // 2. Call ML Service
        let mlClient = MLServiceClient(app: app)
        let mlResult = try await mlClient.processFile(taskID: taskId, filePath: filePath)
        
        // 3. Save Results
        let result = AnalysisResult(
            taskID: taskId,
            slicesData: mlResult.slicesData,
            masksData: mlResult.masksData,
            parameters: mlResult.parameters,
            diagnosis: mlResult.diagnosis
        )
        try await result.save(on: app.db)
        
        // 4. Update status to completed
        try await updateStatus(app: app, taskId: taskId, status: .completed)
        
        app.logger.info("Task \(taskId) completed successfully")
    }
    
    private func updateStatus(app: Application, taskId: UUID, status: TaskStatus, error: String? = nil) async throws {
        guard let task = try await AnalysisTask.find(taskId, on: app.db) else {
            app.logger.error("Task \(taskId) not found during status update")
            return
        }
        
        task.status = status
        if let error = error {
            task.errorMessage = error
        }
        task.updatedAt = Date()
        
        try await task.save(on: app.db)
    }
}

// MARK: - DTOs

struct UploadRequest: Content {
    var file: File
}

struct AnalysisUploadResponse: Content {
    var taskId: UUID
}

struct AnalysisResponse: Content {
    let taskId: UUID
    let status: String
    let errorMessage: String?
    let slicesData: String?
    let masksData: String?
    let parameters: String?
    let diagnosis: String?
    
    init(taskId: UUID, status: String, errorMessage: String? = nil, slicesData: String? = nil, masksData: String? = nil, parameters: String? = nil, diagnosis: String? = nil) {
        self.taskId = taskId
        self.status = status
        self.errorMessage = errorMessage
        self.slicesData = slicesData
        self.masksData = masksData
        self.parameters = parameters
        self.diagnosis = diagnosis
    }
}

struct StatusResponse: Content {
    let taskId: UUID
    let status: String
    let errorMessage: String?
}
