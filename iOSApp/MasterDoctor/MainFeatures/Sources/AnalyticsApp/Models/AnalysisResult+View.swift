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
        GroupBox {
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
                Divider()
                result.tmjRight?.makeView(position: .right)
            }
            .padding()
        }
    }
}
