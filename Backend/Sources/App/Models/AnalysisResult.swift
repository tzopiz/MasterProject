import Vapor
import Fluent

final class AnalysisResult: Model, Content, @unchecked Sendable {
    static let schema = "analysis_results"
    
    @ID(key: .id)
    var id: UUID?
    
    @Parent(key: "task_id")
    var task: AnalysisTask
    
    @OptionalField(key: "slices_data")
    var slicesData: String?  // JSON string
    
    @OptionalField(key: "masks_data")
    var masksData: String?  // JSON string
    
    @OptionalField(key: "parameters")
    var parameters: String?  // JSON string
    
    @OptionalField(key: "diagnosis")
    var diagnosis: String?  // JSON string
    
    @Timestamp(key: "created_at", on: .create)
    var createdAt: Date?
    
    init() { }
    
    init(id: UUID? = nil, taskID: UUID, slicesData: String? = nil, masksData: String? = nil, parameters: String? = nil, diagnosis: String? = nil) {
        self.id = id
        self.$task.id = taskID
        self.slicesData = slicesData
        self.masksData = masksData
        self.parameters = parameters
        self.diagnosis = diagnosis
    }
}

struct CreateAnalysisResult: AsyncMigration {
    func prepare(on database: Database) async throws {
        try await database.schema(AnalysisResult.schema)
            .id()
            .field("task_id", .uuid, .required, .references(AnalysisTask.schema, "id", onDelete: .cascade))
            .field("slices_data", .string)
            .field("masks_data", .string)
            .field("parameters", .string)
            .field("diagnosis", .string)
            .field("created_at", .datetime)
            .create()
    }
    
    func revert(on database: Database) async throws {
        try await database.schema(AnalysisResult.schema).delete()
    }
}

