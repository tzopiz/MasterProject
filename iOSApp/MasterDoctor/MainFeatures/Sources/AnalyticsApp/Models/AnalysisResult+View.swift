//  Created by Dmitrii Korchagin on 25.11.2025.

import SwiftUI

extension AnalysisResult {
    func makeView() -> some View {
        ResultsView(result: self)
    }
}

fileprivate struct ResultsView: View {
    let result: AnalysisResult

    var body: some View {
        VStack(spacing: 16) {
            if let diagnosis = result.diagnosis {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Diagnosis")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .textCase(.uppercase)

                        Text(diagnosis.status)
                            .font(.title3)
                            .fontWeight(.bold)
                            .foregroundColor(diagnosis.status.lowercased().contains("normal") ? .green : .red)
                    }

                    Spacer()

                    if let confidence = diagnosis.confidence {
                        VStack(alignment: .trailing, spacing: 2) {
                            Text("Confidence")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .textCase(.uppercase)

                            Text("\(Int(confidence * 100))%")
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundColor(.primary)
                        }
                    }
                }
            }

            VStack(alignment: .leading, spacing: 15) {
                Text("Detection Results")
                    .font(.headline)

                if let volumeShape = result.volumeShape {
                    HStack {
                        Text("Volume Shape:")
                            .foregroundColor(.secondary)
                        Text("\(volumeShape[0]) × \(volumeShape[1]) × \(volumeShape[2])")
                            .fontWeight(.medium)
                    }
                }

                Divider()
                result.tmjLeft?.makeView(position: .left)
                    .frame(maxWidth: .infinity)
                Divider()
                result.tmjRight?.makeView(position: .right)
                    .frame(maxWidth: .infinity)
            }
        }
    }
}
