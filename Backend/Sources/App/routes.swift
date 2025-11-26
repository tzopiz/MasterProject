import Vapor

func routes(_ app: Application) throws {
    // Health check endpoint
    let healthController = HealthController()
    try app.register(collection: healthController)
    
    // Analysis (DICOM upload and processing)
    let analysisController = AnalysisController()
    try app.register(collection: analysisController)
}

