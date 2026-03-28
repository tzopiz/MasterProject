//  Created by Dmitrii Korchagin on 26.11.2025.

import SwiftUI
import FoundationInternal

/// A thumbnail view for a single DICOM file
public struct DICOMThumbnailView: View {
    let file: DICOMFile
    let isSelected: Bool
    let onTap: () -> Void
    
    @State private var thumbnail: Image?
    
    private let imageConverter = DICOMImageConverter()
    
    public init(file: DICOMFile, isSelected: Bool = false, onTap: @escaping () -> Void) {
        self.file = file
        self.isSelected = isSelected
        self.onTap = onTap
    }
    
    public var body: some View {
        Button(action: onTap) {
            ZStack {
                if let thumbnail = thumbnail {
                    thumbnail
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else {
                    Rectangle()
                        .fill(.quaternary)
                    
                    ProgressView()
                        .progressViewStyle(.circular)
                        .scaleEffect(0.6)
                }
            }
            .frame(width: 60, height: 60)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
            )
        }
        .task {
            await loadThumbnail()
        }
    }
    
    private func loadThumbnail() async {
        if let uiImage = imageConverter.convert(file) {
            await MainActor.run {
                thumbnail = Image(uiImage: uiImage)
            }
        }
    }
}

// MARK: - Series Thumbnail Strip

/// A horizontal strip of DICOM thumbnails for series navigation
public struct DICOMSeriesStripView: View {
    let series: DICOMSeries
    @Binding var selectedIndex: Int
    
    public init(series: DICOMSeries, selectedIndex: Binding<Int>) {
        self.series = series
        self._selectedIndex = selectedIndex
    }
    
    public var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 8) {
                    ForEach(Array(series.files.enumerated()), id: \.offset) { index, file in
                        DICOMThumbnailView(
                            file: file,
                            isSelected: index == selectedIndex
                        ) {
                            withAnimation(.easeOut(duration: 0.15)) {
                                selectedIndex = index
                            }
                        }
                        .id(index)
                    }
                }
                .padding(.horizontal, 12)
            }
            .onChange(of: selectedIndex) { _, newValue in
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo(newValue, anchor: .center)
                }
            }
        }
        .frame(height: 70)
    }
}
