import Vapor

func routes(_ app: Application) throws {
    // Health check endpoint
    let healthController = HealthController()
    try app.register(collection: healthController)
    
    // DICOM upload and processing
    let dicomController = DICOMController()
    try app.register(collection: dicomController)
    
    // Analysis results
    let analysisController = AnalysisController()
    try app.register(collection: analysisController)
}

