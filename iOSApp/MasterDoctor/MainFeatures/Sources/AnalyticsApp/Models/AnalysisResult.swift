//  Created by Dmitrii Korchagin on 22.11.2025.

import Foundation

struct AnalysisResult: Sendable {
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


    let taskId: UUID
    let slices: SlicesData?
    let masks: MasksData?
    let parameters: GeometricParameters?
    let diagnosis: DiagnosisData?
    let tmjLeft: BoundingBox?
    let tmjRight: BoundingBox?
    let volumeShape: [Int]?
    
    static func mock() -> AnalysisResult {
        AnalysisResult(
            taskId: UUID(),
            slices: nil,
            masks: nil,
            parameters: GeometricParameters(
                fossaHeight: 12.5,
                headHeight: 8.4,
                width: 15.2,
                additionalParams: ["Joint Space": 2.1]
            ),
            diagnosis: DiagnosisData(
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
}
