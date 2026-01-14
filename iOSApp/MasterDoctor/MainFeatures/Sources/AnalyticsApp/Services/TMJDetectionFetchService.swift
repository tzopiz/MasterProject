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
        
        // --- MOCK IMPLEMENTATION START ---
        // Simulating upload and processing delay
        do {
            try await Task.sleep(nanoseconds: 1_000_000_000) // 1 sec loading
            statusMessage = "Uploading files..."
            try await Task.sleep(nanoseconds: 1_500_000_000) // 1.5 sec upload
            statusMessage = "Processing (Task: \(uid.uuidString.prefix(8))...)..."
            try await Task.sleep(nanoseconds: 2_000_000_000) // 2 sec processing
            
            self.statusMessage = "✓ Detection Complete!"
            self.isProcessing = false
            return getMockResult(taskId: uid)
        } catch {
             self.isProcessing = false
             throw FetchError.unknown
        }
        // --- MOCK IMPLEMENTATION END ---

//        do {
//            var files: [(filename: String, data: Data)] = []
//
//            for (index, url) in selectedFiles.enumerated() {
//                statusMessage = "Loading file \(index + 1)/\(selectedFiles.count)..."
//
//                guard url.startAccessingSecurityScopedResource() else { continue }
//                defer { url.stopAccessingSecurityScopedResource() }
//
//                let data = try Data(contentsOf: url)
//                files.append((filename: url.lastPathComponent, data: data))
//            }
//
//            statusMessage = "Uploading \(files.count) files..."
//
//            let multipartData = endpoint.createMultipartBodyForSeries(files: files, boundary: uid.uuidString)
//            let uploadResp: UploadResponse = try await networkingService.upload(endpoint, from: multipartData)
//
//            statusMessage = "Processing (Task: \(uploadResp.taskId.uuidString.prefix(8))...)..."
//
//            let result = try await pollForResults(taskId: uploadResp.taskId)
//
//            self.statusMessage = "✓ Detection Complete!"
//            return result
//        } catch {
//            self.statusMessage = "Error: \(error.localizedDescription)"
//        }
//
//        isProcessing = false
//        throw FetchError.unknown
    }
    
    private func getMockResult(taskId: UUID) -> AnalysisResult {
        return AnalysisResult(
            taskId: taskId,
            slices: nil,
            masks: nil,
            parameters: AnalysisResult.GeometricParameters(
                fossaHeight: 12.5,
                headHeight: 8.4,
                width: 15.2,
                additionalParams: ["Joint Space": 2.1]
            ),
            diagnosis: AnalysisResult.DiagnosisData(
                status: "Pathology Detected",
                confidence: 0.92,
                recommendations: ["Consult with specialist", "Additional MRI recommended"],
                disclaimer: "AI generated result. Verification required."
            ),
            tmjLeft: BoundingBox(
                center: [180.0, 230.0, 145.0], // Z, Y, X
                bbox: [140, 190, 105, 220, 270, 185] // Z1, Y1, X1, Z2, Y2, X2
            ),
            tmjRight: BoundingBox(
                center: [175.0, 228.0, 430.0], // Z, Y, X
                bbox: [135, 188, 390, 215, 268, 470] // Z1, Y1, X1, Z2, Y2, X2
            ),
            volumeShape: [350, 512, 512]
        )
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
