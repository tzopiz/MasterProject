//  Created by Dmitrii Korchagin on 25.11.2025.

import Foundation
import CoreNetwork
import FoundationInternal

actor TMJDetectionFetchService: Sendable {
    enum FetchError: Error {
        case notResponse
        case emptySelectedFiles
        case statusProcessingFailed(message: String?)
        case timeout
        case unknown
    }

    private let networkingService: any NetworkingServiceProtocol
    private let decoder: any JSONDecoderProtocol

    private(set) var isProcessing = false
    private var statusMessage = "" // AsyncStream

    init(
        networkingService: any NetworkingServiceProtocol,
        decoder: any JSONDecoderProtocol,
    ) {
        self.networkingService = networkingService
        self.decoder = decoder
    }

    func runAnalysis(_ uid: UUID, endpoint: any Endpoint, selectedFiles: [URL]) async throws -> AnalysisResult {
        guard !selectedFiles.isEmpty else {
            throw FetchError.emptySelectedFiles
        }

        self.isProcessing = true
        statusMessage = "Loading files..."

        do {
            var files: [(filename: String, data: Data)] = []

            for (index, url) in selectedFiles.enumerated() {
                statusMessage = "Loading file \(index + 1)/\(selectedFiles.count)..."

                guard url.startAccessingSecurityScopedResource() else { continue }
                defer { url.stopAccessingSecurityScopedResource() }

                let data = try Data(contentsOf: url)
                files.append((filename: url.lastPathComponent, data: data))
            }

            statusMessage = "Uploading \(files.count) files..."

//            let endpoint = AnalysisEndpoint.uploadSeries(files: files, boundary: boundary)
            let multipartData = endpoint.createMultipartBodyForSeries(files: files, boundary: uid.uuidString)
            let uploadResp: UploadResponse = try await networkingService.upload(endpoint, from: multipartData)

            statusMessage = "Processing (Task: \(uploadResp.taskId.uuidString.prefix(8))...)..."

            let result = try await pollForResults(taskId: uploadResp.taskId)

            self.statusMessage = "✓ Detection Complete!"
            return result
        } catch {
            self.statusMessage = "Error: \(error.localizedDescription)"
        }

        isProcessing = false
        throw FetchError.unknown
    }

    private func pollForResults(taskId: UUID) async throws -> AnalysisResult {
        let maxAttempts = 10
        let interval: UInt64 = 2_000_000_000

        for attempt in 0..<maxAttempts {
            let statusResp: StatusResponse = try await networkingService.request(
                AnalysisEndpoint.status(taskId: taskId)
            )
            switch statusResp.status {
            case "completed":
                let rawResult: RawAnalysisResponse = try await networkingService.request(
                    AnalysisEndpoint.result(taskId: taskId)
                )
                return convert(from: rawResult)

            case "failed":
                throw FetchError.statusProcessingFailed(message: statusResp.errorMessage)

            default:
                statusMessage = "Processing... (\(attempt + 1)/\(maxAttempts))"
                try await Task.sleep(nanoseconds: interval)
            }
        }

        throw FetchError.timeout
    }

    private func convert(from raw: RawAnalysisResponse) -> AnalysisResult {
        AnalysisResult(
            taskId: raw.taskId,
            slices: decoder.decode(AnalysisResult.SlicesData.self, from: raw.slicesData),
            masks: decoder.decode(AnalysisResult.MasksData.self, from: raw.masksData),
            parameters: decoder.decode(AnalysisResult.GeometricParameters.self, from: raw.parameters),
            diagnosis: decoder.decode(AnalysisResult.DiagnosisData.self, from: raw.diagnosis),
            tmjLeft: decoder.decode(BoundingBox.self, from: raw.tmjLeft),
            tmjRight: decoder.decode(BoundingBox.self, from: raw.tmjRight),
            volumeShape: raw.volumeShape
        )
    }
}
