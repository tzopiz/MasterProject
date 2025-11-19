import Vapor
import Fluent

final class AnalysisTask: Model, Content, @unchecked Sendable {
    static let schema = "analysis_tasks"
    
    @ID(key: .id)
    var id: UUID?
    
    @Field(key: "dicom_filename")
    var dicomFilename: String
    
    @Field(key: "dicom_path")
    var dicomPath: String
    
    @Field(key: "status")
    var status: TaskStatus
    
    @OptionalField(key: "error_message")
    var errorMessage: String?
    
    @Timestamp(key: "created_at", on: .create)
    var createdAt: Date?
    
    @Timestamp(key: "updated_at", on: .update)
    var updatedAt: Date?
    
    init() { }
    
    init(id: UUID? = nil, dicomFilename: String, dicomPath: String, status: TaskStatus = .pending) {
        self.id = id
        self.dicomFilename = dicomFilename
        self.dicomPath = dicomPath
        self.status = status
    }
}

enum TaskStatus: String, Codable {
    case pending
    case processing
    case completed
    case failed
}

struct CreateAnalysisTask: AsyncMigration {
    func prepare(on database: Database) async throws {
        try await database.schema(AnalysisTask.schema)
            .id()
            .field("dicom_filename", .string, .required)
            .field("dicom_path", .string, .required)
            .field("status", .string, .required)
            .field("error_message", .string)
            .field("created_at", .datetime)
            .field("updated_at", .datetime)
            .create()
    }
    
    func revert(on database: Database) async throws {
        try await database.schema(AnalysisTask.schema).delete()
    }
}

