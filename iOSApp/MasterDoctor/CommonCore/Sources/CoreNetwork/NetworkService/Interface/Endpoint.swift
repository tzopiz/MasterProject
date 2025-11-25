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
    typealias File = (filename: String, data: Data)

    var url: URL { get }
    var path: String { get }
    var method: HTTPMethod { get }
    var headers: Headers { get }
    var body: Data? { get }
    var queryItems: QueryItems { get }

    func createMultipartBody(filename: String, fileData: Data, boundary: String) -> Data
    func createMultipartBodyForSeries(files: [File], boundary: String) -> Data
}

public extension Endpoint {
    var headers: Headers { nil }
    var body: Data? { nil }
    var queryItems: QueryItems { nil }

    func createMultipartBody(filename: String, fileData: Data, boundary: String) -> Data {
        var data = Data()
        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        data.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
        data.append(fileData)
        data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        return data
    }

    func createMultipartBodyForSeries(files: [(filename: String, data: Data)], boundary: String) -> Data {
        var data = Data()

        for file in files {
            data.append("--\(boundary)\r\n".data(using: .utf8)!)
            data.append("Content-Disposition: form-data; name=\"files\"; filename=\"\(file.filename)\"\r\n".data(using: .utf8)!)
            data.append("Content-Type: application/dicom\r\n\r\n".data(using: .utf8)!)
            data.append(file.data)
            data.append("\r\n".data(using: .utf8)!)
        }

        data.append("--\(boundary)--\r\n".data(using: .utf8)!)
        return data
    }
}
