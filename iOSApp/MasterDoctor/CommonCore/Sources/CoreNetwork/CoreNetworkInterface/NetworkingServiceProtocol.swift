//  Created by Dmitrii Korchagin on 22.11.2025.

import Foundation

public protocol NetworkingServiceProtocol: Sendable {
    /// Выполнить стандартный JSON запрос
    func request<T: Decodable & Sendable>(_ endpoint: Endpoint) async throws -> T

    /// Загрузить данные (Multipart или Raw)
    func upload<T: Decodable & Sendable>(_ endpoint: Endpoint, from data: Data) async throws -> T
}
