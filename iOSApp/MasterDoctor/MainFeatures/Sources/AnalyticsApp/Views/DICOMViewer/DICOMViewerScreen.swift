//  Created by Dmitrii Korchagin on 26.11.2025.

import SwiftUI
import FoundationInternal

/// Full-screen DICOM viewer with all controls
public struct DICOMViewerScreen: View {
    @Environment(\.dismiss) private var dismiss
    
    @State private var currentSliceIndex: Int = 0
    @State private var showThumbnailStrip = false
    
    private let series: DICOMSeries
    
    public init(series: DICOMSeries) {
        self.series = series
    }
    
    public var body: some View {
        ZStack {
            // Main viewer
            DICOMViewerView(series: series)
            
            // Navigation bar overlay
            VStack {
                navigationBar

                Spacer()
                
                if showThumbnailStrip {
                    DICOMSeriesStripView(
                        series: series,
                        selectedIndex: $currentSliceIndex
                    )
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
        }
        .ignoresSafeArea()
        .navigationBarHidden(true)
        .statusBar(hidden: true)
    }
    
    // MARK: - Navigation Bar
    
    private var navigationBar: some View {
        HStack {
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.title3)
                    .fontWeight(.medium)
                    .foregroundStyle(.primary)
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial, in: Circle())
            }
            
            Spacer()
            
            // Series info
            if let modality = series.modality {
                HStack(spacing: 6) {
                    Image(systemName: "doc.viewfinder")
                        .font(.caption)
                    Text(modality)
                        .font(.caption)
                        .fontWeight(.semibold)
                }
                .foregroundStyle(.primary)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
            }
            
            Spacer()
            
            Button {
                withAnimation(.easeInOut(duration: 0.25)) {
                    showThumbnailStrip.toggle()
                }
            } label: {
                Image(systemName: showThumbnailStrip ? "rectangle.grid.1x2.fill" : "rectangle.grid.1x2")
                    .font(.title3)
                    .foregroundStyle(.primary)
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial, in: Circle())
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 50) // Safe area
    }
}

// MARK: - Preview

#if DEBUG
#Preview {
    DICOMViewerScreen(series: .mock(sliceCount: 12, pattern: .skull))
}
#endif
