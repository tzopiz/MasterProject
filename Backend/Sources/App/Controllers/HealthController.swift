import Vapor

struct HealthController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        routes.get("health", use: health)
    }
    
    func health(req: Request) async throws -> HealthResponse {
        return HealthResponse(status: "ok", service: "vapor-backend", timestamp: Date())
    }
}

struct HealthResponse: Content {
    let status: String
    let service: String
    let timestamp: Date
}

