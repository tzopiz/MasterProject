//  Created by Dmitrii Korchagin on 22.11.2025.

import Foundation

struct UploadResponse: Codable, Sendable {
    let taskId: UUID
}

struct StatusResponse: Codable, Sendable {
    let taskId: UUID
    let status: String
    let errorMessage: String?
}

struct RawAnalysisResponse: Codable, Sendable {
    let taskId: UUID
    let status: String
    let slicesData: String?
    let masksData: String?
    let parameters: String?
    let diagnosis: String?
    let tmjLeft: String?
    let tmjRight: String?
    let volumeShape: [Int]?
    let createdAt: String?
    let updatedAt: String?
}
