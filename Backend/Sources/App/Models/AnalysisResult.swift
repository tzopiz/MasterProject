import Vapor
import Fluent

final class AnalysisResult: Model, Content, @unchecked Sendable {
    static let schema = "analysis_results"
    
    @ID(key: .id)
    var id: UUID?
    
    @Parent(key: "task_id")
    var task: AnalysisTask
    
    // Bounding Boxes (stored as JSON string)
    @Field(key: "tmj_left")
    var tmjLeft: String?
    
    @Field(key: "tmj_right")
    var tmjRight: String?
    
    // Volume dimensions for visualization
    @OptionalField(key: "volume_shape")
    var volumeShape: [Int]?
    
    // Legacy fields (kept but optional/unused for now)
    @OptionalField(key: "slices_data")
    var slicesData: String?
    
    @OptionalField(key: "masks_data")
    var masksData: String?
    
    @OptionalField(key: "parameters")
    var parameters: String?
    
    @OptionalField(key: "diagnosis")
    var diagnosis: String?
    
    init() { }
    
    init(id: UUID? = nil, taskID: AnalysisTask.IDValue, tmjLeft: String? = nil, tmjRight: String? = nil, volumeShape: [Int]? = nil) {
        self.id = id
        self.$task.id = taskID
        self.tmjLeft = tmjLeft
        self.tmjRight = tmjRight
        self.volumeShape = volumeShape
    }
    
    // Legacy init
    init(id: UUID? = nil, taskID: AnalysisTask.IDValue, slicesData: String?, masksData: String?, parameters: String?, diagnosis: String?) {
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
            .field("task_id", .uuid, .required, .references("analysis_tasks", "id"))
            .field("tmj_left", .string)  // JSON String
            .field("tmj_right", .string) // JSON String
            .field("volume_shape", .array(of: .int)) // [depth, height, width]
            .field("slices_data", .string) // Legacy
            .field("masks_data", .string)  // Legacy
            .field("parameters", .string)  // Legacy
            .field("diagnosis", .string)   // Legacy
            .create()
    }
    
    func revert(on database: Database) async throws {
        try await database.schema(AnalysisResult.schema).delete()
    }
}
