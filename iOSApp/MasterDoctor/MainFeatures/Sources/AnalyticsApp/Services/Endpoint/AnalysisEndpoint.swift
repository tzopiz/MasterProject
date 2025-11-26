//  Created by Dmitrii Korchagin on 22.11.2025.

import Foundation
import CoreNetwork

enum AnalysisEndpoint: Endpoint {
    case upload(filename: String, data: Data, boundary: String)
    case uploadSeries(files: [(filename: String, data: Data)], boundary: String)
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
        case .upload, .uploadSeries: "analysis"
        case .status(let taskId): "analysis/\(taskId.uuidString)/status"
        case .result(let taskId): "analysis/\(taskId.uuidString)"
        }
    }

    var method: HTTPMethod {
        switch self {
        case .upload, .uploadSeries: .post
        case .status, .result: .get
        }
    }

    var headers: Headers {
        switch self {
        case .upload(_, _, let boundary), .uploadSeries(_, let boundary):
            ["Content-Type": "multipart/form-data; boundary=\(boundary)"]
        default:
            nil
        }
    }
}

fileprivate let baseUrlString = "http://localhost:8080/api"
