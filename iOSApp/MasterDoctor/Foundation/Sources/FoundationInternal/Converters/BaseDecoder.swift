//  Created by Dmitrii Korchagin on 25.11.2025.

import Foundation

public protocol JSONDecoderProtocol: Sendable {
    func decode<T: Decodable>(_ type: T.Type, from data: Data?) throws -> T
    func decode<T: Decodable>(_ type: T.Type, from str: String?) throws -> T

    func decode<T: Decodable>(_ type: T.Type, from data: Data?) -> T?
    func decode<T: Decodable>(_ type: T.Type, from str: String?) -> T?
}

public struct JSONDecoderService: JSONDecoderProtocol {
    enum JSONDecoderError: Error {
        case dataCorrupted
    }
    private let defaultDecoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    public init() {}

    public func decode<T: Decodable>(_ type: T.Type, from data: Data?) throws -> T {
        guard let data else { throw JSONDecoderError.dataCorrupted }
        return try defaultDecoder.decode(type, from: data)
    }

    public func decode<T: Decodable>(_ type: T.Type, from str: String?) throws -> T {
        guard let str, let data = str.data(using: .utf8) else {
            throw JSONDecoderError.dataCorrupted
        }
        return try decode(type, from: data)
    }

    public func decode<T: Decodable>(_ type: T.Type, from data: Data?) -> T? {
        guard let data else { return nil }
        return try? defaultDecoder.decode(type, from: data)
    }

    public func decode<T: Decodable>(_ type: T.Type, from str: String?) -> T? {
        guard let str, let data = str.data(using: .utf8) else { return nil }
        return try? decode(type, from: data)
    }
}
