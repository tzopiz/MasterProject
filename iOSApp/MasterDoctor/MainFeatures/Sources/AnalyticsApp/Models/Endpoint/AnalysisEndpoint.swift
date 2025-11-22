//  Created by Dmitrii Korchagin on 22.11.2025.

import Foundation
import CoreNetworkInterface

enum AnalysisEndpoint: Endpoint {
    case upload(filename: String, data: Data, boundary: String)
    case status(taskId: UUID)
    case result(taskId: UUID)

    var url: URL {
        guard let url = URL(string: baseUrlString) else {
            fatalError("Invalid Base URL")
        }
        return url
    }

    var path: String {
        switch self {
        case .upload: "analysis"
        case .status(let taskId): "analysis/\(taskId.uuidString)/status"
        case .result(let taskId): "analysis/\(taskId.uuidString)"
        }
    }

    var method: HTTPMethod {
        switch self {
        case .upload: .post
        case .status, .result: .get
        }
    }

    var headers: Headers {
        switch self {
        case .upload(_, _, let boundary):
            ["Content-Type": "multipart/form-data; boundary=\(boundary)"]
        default:
            nil
        }
    }

    var body: Data? { nil }
}

extension AnalysisEndpoint {
    static func createMultipartBody(filename: String, fileData: Data, boundary: String) -> Data {
        var data = Data()
        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        data.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
        data.append(fileData)
        data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        return data
    }
}

private let baseUrlString = "http://localhost:8080/api"
