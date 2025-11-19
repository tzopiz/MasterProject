import Vapor
import Fluent
import FluentSQLiteDriver

public func configure(_ app: Application) async throws {
    // Configure maximum file upload size (500 MB for large DICOM files)
    app.routes.defaultMaxBodySize = "500mb"
    
    // Configure SQLite database
    app.databases.use(.sqlite(.file("db.sqlite")), as: .sqlite)
    
    // Add migrations
    app.migrations.add(CreateAnalysisTask())
    app.migrations.add(CreateAnalysisResult())
    
    // Auto-migrate database
    try await app.autoMigrate()
    
    // Register routes
    try routes(app)
    
    app.logger.info("Backend configured successfully")
}

