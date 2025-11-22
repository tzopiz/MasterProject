//  Created by Dmitrii Korchagin on 22.11.2025.

import Foundation
import CoreNetworkInterface

public final class NetworkingService: NetworkingServiceProtocol {
    private let session: URLSession
    
    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func request<T: Decodable & Sendable>(_ endpoint: Endpoint) async throws -> T {
        let request = try buildRequest(for: endpoint)
        let (data, response) = try await session.data(for: request)
        try validate(response: response)
        return try decoder.decode(T.self, from: data)
    }

    public func upload<T: Decodable & Sendable>(_ endpoint: Endpoint, from uploadData: Data) async throws -> T {
        let request = try buildRequest(for: endpoint)
        let (data, response) = try await session.upload(for: request, from: uploadData)
        try validate(response: response)
        return try decoder.decode(T.self, from: data)
    }
}

// MARK: - Private methods

extension NetworkingService {
    private func buildRequest(for endpoint: Endpoint) throws -> URLRequest {
        var url = endpoint.url.appendingPathComponent(endpoint.path)

        if let queryItems = endpoint.queryItems, !queryItems.isEmpty {
            guard var components = URLComponents(url: url, resolvingAgainstBaseURL: true) else {
                throw URLError(.badURL)
            }
            components.queryItems = queryItems
            guard let newUrl = components.url else {
                throw URLError(.badURL)
            }
            url = newUrl
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = endpoint.method.rawValue
        request.allHTTPHeaderFields = endpoint.headers
        request.httpBody = endpoint.body
        
        return request
    }
    
    private func validate(response: URLResponse) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        
        guard (200...299).contains(httpResponse.statusCode) else {
            throw URLError(.badServerResponse)
        }
    }
}
