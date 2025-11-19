import Vapor
import Fluent

struct AnalysisController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        let analysis = routes.grouped("api", "analysis")
        analysis.get(":taskId", use: getResult)
        analysis.get(":taskId", "status", use: getStatus)
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

