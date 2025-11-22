//  Created by Dmitrii Korchagin on 22.11.2025.

import CoreNetworkInterface
import Foundation

public struct MockNetworkingService: NetworkingServiceProtocol {
    enum MockServerError: Error {
        case usedMockServer
    }

    public init() {}

    public func request<T: Decodable & Sendable>(_: any CoreNetworkInterface.Endpoint) async throws -> T {
        throw MockServerError.usedMockServer
    }
    
    public func upload<T: Decodable & Sendable>(_: any CoreNetworkInterface.Endpoint, from _: Data) async throws -> T {
        throw MockServerError.usedMockServer
    }
}
