//  Created by Dmitrii Korchagin on 22.11.2025.

import SwiftUI
import CoreNetworkInterface
import CoreNetworkImpl
import CoreSwiftUI

public struct AnalysisResultView: View {
    @State private var analysisResult: AnalysisResult?
    @State private var statusMessage: String = "Ready"
    @State private var isProcessing = false

    @Environment(\.deps) private var deps
    
    public init() {}

    public var body: some View {
        List {
            Section {
                if let result = analysisResult {
                    Text("Analysis Complete!")
                        .font(.title)
                        .foregroundColor(.green)

                    if let params = result.parameters {
                        Text("Fossa Height: \(params.fossaHeight ?? 0, specifier: "%.3f") mm")
                        Text("Head Height: \(params.headHeight ?? 0, specifier: "%.3f") mm")
                    }

                    if let diagnosis = result.diagnosis {
                        Text("Diagnosis: \(diagnosis.status)")
                            .bold()
                    }
                } else {
                    Text(statusMessage)
                }

                if isProcessing {
                    ProgressView()
                }
            } header: {
                if let taskId = analysisResult?.taskId {
                    Text(taskId.uuidString)
                }
            }
        }
        .refreshable(action: runAnalysis)
        .task {
            await runAnalysis()
        }
    }

    @Sendable
    func runAnalysis() async {
        isProcessing = true
        statusMessage = "Uploading..."

        do {
            let url = URL(fileURLWithPath: dcmPath)
            let fileData = try Data(contentsOf: url)
            let boundary = UUID().uuidString
            let filename = url.lastPathComponent

            let multipartData = AnalysisEndpoint.createMultipartBody(filename: filename, fileData: fileData, boundary: boundary)
            let endpoint = AnalysisEndpoint.upload(filename: filename, data: multipartData, boundary: boundary)

            let uploadResp: UploadResponse = try await deps.networkingService.upload(endpoint, from: multipartData)
            statusMessage = "Processing (Task ID: \(uploadResp.taskId))..."

            let result = try await pollForResults(taskId: uploadResp.taskId)

            self.analysisResult = result
            self.statusMessage = "Done"

        } catch {
            self.statusMessage = "Error: \(error.localizedDescription)"
        }

        isProcessing = false
    }
    
    private func pollForResults(taskId: UUID) async throws -> AnalysisResult {
        let maxAttempts = 150
        let interval: UInt64 = 2_000_000_000
        
        for _ in 0..<maxAttempts {
            let statusResp: StatusResponse = try await deps.networkingService.request(AnalysisEndpoint.status(taskId: taskId))

            switch statusResp.status {
            case "completed":
                let rawResult: RawAnalysisResponse = try await deps.networkingService.request(AnalysisEndpoint.result(taskId: taskId))
                return parseRawResult(rawResult)
                
            case "failed":
                throw NSError(domain: "App", code: -1, userInfo: [NSLocalizedDescriptionKey: statusResp.errorMessage ?? "Error"])
                
            default:
                try await Task.sleep(nanoseconds: interval)
            }
        }
        throw NSError(domain: "App", code: -2, userInfo: [NSLocalizedDescriptionKey: "Timeout"])
    }
    
    private func parseRawResult(_ raw: RawAnalysisResponse) -> AnalysisResult {
        // Helper to decode inner JSONs (since we moved it out of Service)
        func decode<T: Decodable>(_ str: String?, to type: T.Type) -> T? {
            guard let str, let data = str.data(using: .utf8) else { return nil }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try? decoder.decode(T.self, from: data)
        }
        
        return AnalysisResult(
            taskId: raw.taskId,
            slices: decode(raw.slicesData, to: SlicesData.self),
            masks: decode(raw.masksData, to: MasksData.self),
            parameters: decode(raw.parameters, to: GeometricParameters.self),
            diagnosis: decode(raw.diagnosis, to: DiagnosisData.self)
        )
    }
}

private let dcmPath = "/Users/tzopiz/Developer/MasterProject/iOSApp/MasterDoctor/MasterDoctor/example.dcm"
