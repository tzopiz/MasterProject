//  Created by Dmitrii Korchagin on 25.11.2025.

import CoreSwiftUI
import SwiftUI

struct BoundingBox: Codable, Sendable {
    let center: [Double]  // [z, y, x]
    let bbox: [Int]       // [z1, y1, x1, z2, y2, x2]
}

extension BoundingBox {
    enum Position: String {
        case left = "Left TMJ"
        case right = "Right TMJ"
    }

    func makeView(position: Position) -> some View {
        TMJCoordinateView(bbox: self, position: position)
    }
}

fileprivate struct TMJCoordinateView: View {
    let bbox: BoundingBox
    let position: BoundingBox.Position

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(position.rawValue)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.blue)

            VStack(alignment: .leading, spacing: 4) {
                Text("Center (Z, Y, X):")
                    .font(.caption)
                    .foregroundColor(.secondary)

                HStack(spacing: 12) {
                    CoordinateChip(label: "Z", value: bbox.center[0])
                    CoordinateChip(label: "Y", value: bbox.center[1])
                    CoordinateChip(label: "X", value: bbox.center[2])
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Bounding Box:")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text("[\(bbox.bbox[0]), \(bbox.bbox[1]), \(bbox.bbox[2])] → [\(bbox.bbox[3]), \(bbox.bbox[4]), \(bbox.bbox[5])]")
                    .font(.caption)
                    .fontDesign(.monospaced)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.blue.opacity(0.05))
        .cornerRadius(8)
    }
}

fileprivate struct CoordinateChip: View {
    let label: String
    let value: Double

    var body: some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(String(format: "%.1f", value))
                .font(.caption)
                .fontWeight(.medium)
                .fontDesign(.monospaced)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(6)
    }
}
