//  Created by Dmitrii Korchagin on 22.11.2025.

import Foundation

public enum HTTPMethod: String {
    case get = "GET"
    case post = "POST"
    case put = "PUT"
    case delete = "DELETE"
}

public protocol Endpoint: Sendable {
    typealias Headers = [String: String]?
    typealias QueryItems = [URLQueryItem]?

    var url: URL { get }
    var path: String { get }
    var method: HTTPMethod { get }
    var headers: Headers { get }
    var body: Data? { get }
    var queryItems: QueryItems { get }
}

public extension Endpoint {
    var headers: Headers { nil }
    var body: Data? { nil }
    var queryItems: QueryItems { nil }
}
