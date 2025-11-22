//  Created by Dmitrii Korchagin on 22.11.2025.

import Foundation

struct AnalysisResult: Sendable {
    let taskId: UUID
    let slices: SlicesData?
    let masks: MasksData?
    let parameters: GeometricParameters?
    let diagnosis: DiagnosisData?
}

struct SlicesData: Codable, Sendable {
    let orthogonal: [String]?
    let sagittal: [String]?
    let frontal: [String]?
}

struct MasksData: Codable, Sendable {
    let orthogonal: [String]?
    let sagittal: [String]?
    let frontal: [String]?
}

struct GeometricParameters: Codable, Sendable {
    let fossaHeight: Double?
    let headHeight: Double?
    let width: Double?
    let additionalParams: [String: Double]?
}

struct DiagnosisData: Codable, Sendable {
    let status: String
    let confidence: Double?
    let recommendations: [String]?
    let disclaimer: String?
}
