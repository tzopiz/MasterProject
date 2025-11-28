//  Created by Dmitrii Korchagin on 26.11.2025.

import SwiftUI
import FoundationInternal

/// A view for displaying and navigating DICOM image series
public struct DICOMViewerView: View {
    @State private var currentSliceIndex: Int = 0
    @State private var currentImage: Image?
    @State private var isLoading = true
    @State private var errorMessage: String?
    @State private var diagnosticInfo: DICOMDiagnosticInfo?
    @State private var showDiagnostics = false
    
    @State private var windowCenter: Double = 40
    @State private var windowWidth: Double = 400
    @State private var showControls = true
    @State private var selectedPreset: DICOMWindowPreset = .softTissue
    
    private let series: DICOMSeries
    private let imageConverter = DICOMImageConverter()
    
    public init(series: DICOMSeries) {
        self.series = series
    }
    
    public var body: some View {
        GeometryReader { _ in
            ZStack {
                if let image = currentImage {
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .gesture(sliceNavigationGesture)
                } else if isLoading {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .tint(.primary)
                        .scaleEffect(1.5)
                } else {
                    errorView
                        .safeAreaInset(edge: .top) {
                            Color.clear.frame(height: 0)
                        }
                }

                // Overlay Controls
                if showControls && currentImage != nil {
                    VStack {
                        topInfoBar
                        Spacer()
                        bottomControls
                            .padding(.bottom, 32)
                    }
                    .transition(.opacity)
                }
            }
        }
        .onTapGesture {
            if currentImage != nil {
                withAnimation(.easeInOut(duration: 0.2)) {
                    showControls.toggle()
                }
            }
        }
        .task {
            await loadCurrentSlice()
        }
        .onChange(of: currentSliceIndex) {
            Task {
                await loadCurrentSlice()
            }
        }
        .onChange(of: windowCenter) {
            Task {
                await loadCurrentSlice()
            }
        }
        .onChange(of: windowWidth) {
            Task {
                await loadCurrentSlice()
            }
        }
        .sheet(isPresented: $showDiagnostics) {
            diagnosticsSheet
        }
    }
    
    // MARK: - Error View
    
    private var errorView: some View {
        ScrollView {
            VStack(spacing: 16) {
                Spacer()
                    .frame(height: 40)
                
                Image(systemName: errorIcon)
                    .font(.system(size: 48))
                    .foregroundStyle(errorColor)
                
                Text(errorTitle)
                    .font(.headline)
                    .foregroundStyle(.primary)
                
                if let error = errorMessage {
                    Text(error)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }
                
                // Quick diagnostic summary
                if let info = diagnosticInfo {
                    VStack(alignment: .leading, spacing: 6) {
                        diagnosticRow("Dimensions", "\(info.dimensions)\(info.hasExplicitDimensions ? "" : " ⚠️")")
                        diagnosticRow("Bits", "\(info.bitsAllocated) / \(info.bitsStored)")
                        diagnosticRow("Pixel Data", formatBytes(info.pixelDataSize))
                        diagnosticRow("Expected", formatBytes(info.expectedDataSize))
                        diagnosticRow("DICM Prefix", info.hasDICMPrefix ? "✓" : "✗")
                        diagnosticRow("Compressed", info.isCompressed ? "⚠️ Yes" : "No")
                        
                        // Show first warning if any
                        if let firstWarning = info.warnings.first {
                            Divider()
                            Text("⚠️ \(firstWarning)")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                    }
                    .font(.caption)
                    .padding(12)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                }
                
                HStack(spacing: 16) {
                    Button {
                        showDiagnostics = true
                    } label: {
                        Label("Details", systemImage: "info.circle")
                            .font(.subheadline)
                    }
                    .buttonStyle(.bordered)
                    .tint(.accentColor)
                    
                    if series.sliceCount > 1 {
                        Button {
                            tryNextSlice()
                        } label: {
                            Label("Try Next", systemImage: "arrow.right")
                                .font(.subheadline)
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .padding(.top, 8)
            }
            .padding()
            .frame(maxWidth: .infinity)
        }
    }
    
    private var errorIcon: String {
        guard let info = diagnosticInfo else {
            return "exclamationmark.triangle.fill"
        }
        if info.isCompressed {
            return "lock.fill"
        }
        if !info.hasDICMPrefix {
            return "doc.questionmark.fill"
        }
        return "exclamationmark.triangle.fill"
    }
    
    private var errorColor: Color {
        guard let info = diagnosticInfo else {
            return .orange
        }
        if info.isCompressed {
            return .purple
        }
        return .orange
    }
    
    private var errorTitle: String {
        guard let info = diagnosticInfo else {
            return "Failed to Load Image"
        }
        if info.isCompressed {
            return "Compressed DICOM"
        }
        if !info.hasExplicitDimensions {
            return "Invalid DICOM Structure"
        }
        if info.pixelDataSize == 0 {
            return "No Pixel Data"
        }
        return "Failed to Load Image"
    }
    
    private func diagnosticRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .foregroundStyle(.primary)
        }
    }
    
    // MARK: - Diagnostics Sheet
    
    private var diagnosticsSheet: some View {
        NavigationStack {
            List {
                if let info = diagnosticInfo {
                    Section("Image Properties") {
                        LabeledContent("Dimensions", value: info.dimensions)
                        if !info.hasExplicitDimensions {
                            Text("⚠️ Dimensions not found in file")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                        LabeledContent("Bits Allocated", value: "\(info.bitsAllocated)")
                        LabeledContent("Bits Stored", value: "\(info.bitsStored)")
                        LabeledContent("Pixel Representation", value: info.pixelRepresentation)
                        LabeledContent("Photometric", value: info.photometricInterpretation)
                        LabeledContent("Modality", value: info.modality ?? "Unknown")
                    }
                    
                    Section("DICOM Format") {
                        LabeledContent("DICM Prefix", value: info.hasDICMPrefix ? "Yes ✓" : "No ✗")
                        LabeledContent("Compressed", value: info.isCompressed ? "Yes ⚠️" : "No")
                        if let ts = info.transferSyntax {
                            LabeledContent("Transfer Syntax", value: ts)
                        }
                    }
                    
                    Section("Data") {
                        LabeledContent("Pixel Data Size", value: formatBytes(info.pixelDataSize))
                        LabeledContent("Expected Size", value: formatBytes(info.expectedDataSize))
                        
                        let difference = info.pixelDataSize - info.expectedDataSize
                        if difference != 0 {
                            HStack {
                                Text("Difference")
                                Spacer()
                                Text("\(difference > 0 ? "+" : "")\(formatBytes(abs(difference)))")
                                    .foregroundStyle(difference < 0 ? .red : .orange)
                            }
                        }
                        
                        LabeledContent("Has Window Settings", value: info.hasWindowSettings ? "Yes" : "No")
                    }
                    
                    if !info.warnings.isEmpty {
                        Section("Warnings") {
                            ForEach(info.warnings, id: \.self) { warning in
                                HStack {
                                    Image(systemName: "exclamationmark.triangle.fill")
                                        .foregroundStyle(.orange)
                                        .font(.caption)
                                    Text(warning)
                                        .font(.caption)
                                }
                            }
                        }
                    }
                    
                    if let error = errorMessage {
                        Section("Error") {
                            Text(error)
                                .foregroundStyle(.red)
                        }
                    }
                }
                
                Section("File Info") {
                    if currentSliceIndex < series.files.count {
                        let file = series.files[currentSliceIndex]
                        LabeledContent("Filename", value: file.url.lastPathComponent)
                    }
                    LabeledContent("Slice", value: "\(currentSliceIndex + 1) of \(series.sliceCount)")
                }
            }
            .navigationTitle("DICOM Diagnostics")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        showDiagnostics = false
                    }
                }
            }
        }
    }
    
    private func formatBytes(_ bytes: Int) -> String {
        if bytes >= 1_000_000 {
            return String(format: "%.1f MB", Double(bytes) / 1_000_000)
        } else if bytes >= 1_000 {
            return String(format: "%.1f KB", Double(bytes) / 1_000)
        } else {
            return "\(bytes) bytes"
        }
    }
    
    // MARK: - Top Info Bar

    private var topInfoBar: some View {
        HStack {
            Spacer()

            VStack(alignment: .leading, spacing: 4) {
                if let modality = series.modality {
                    Text(modality)
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.tint)
                }

                if let description = series.seriesDescription {
                    Text(description)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 4) {
                Text("Slice \(currentSliceIndex + 1) / \(series.sliceCount)")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.primary)

                if let metadata = currentMetadata {
                    Text("\(metadata.columns) × \(metadata.rows)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
    }
    
    // MARK: - Bottom Controls
    
    private var bottomControls: some View {
        VStack(spacing: 16) {
            // Slice slider
            VStack(spacing: 8) {
                HStack {
                    Text("Slice")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    Spacer()
                    
                    Text("\(currentSliceIndex + 1)")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(.tint)
                }
                
                Slider(
                    value: Binding(
                        get: { Double(currentSliceIndex) },
                        set: { currentSliceIndex = Int($0) }
                    ),
                    in: 0...Double(max(0, series.sliceCount - 1)),
                    step: 1
                )
                .tint(.accentColor)
            }
            
            // Window/Level controls
            VStack(spacing: 12) {
                // Presets
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(DICOMWindowPreset.allCases.filter { $0 != .custom }, id: \.self) { preset in
                            Button {
                                selectedPreset = preset
                                windowCenter = preset.windowCenter
                                windowWidth = preset.windowWidth
                            } label: {
                                Text(preset.rawValue)
                                    .font(.caption)
                                    .fontWeight(selectedPreset == preset ? .semibold : .regular)
                            }
                            .buttonStyle(.bordered)
                            .tint(selectedPreset == preset ? .accentColor : .secondary)
                        }
                    }
                    .padding(.horizontal, 4)
                }
                
                // Window Center
                HStack {
                    Text("W/C")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .frame(width: 30, alignment: .leading)
                    
                    Slider(value: $windowCenter, in: -1000...1000)
                        .tint(.secondary)
                    
                    Text("\(Int(windowCenter))")
                        .font(.caption2)
                        .foregroundStyle(.primary)
                        .frame(width: 45, alignment: .trailing)
                }
                
                // Window Width
                HStack {
                    Text("W/W")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .frame(width: 30, alignment: .leading)
                    
                    Slider(value: $windowWidth, in: 1...4000)
                        .tint(.secondary)
                    
                    Text("\(Int(windowWidth))")
                        .font(.caption2)
                        .foregroundStyle(.primary)
                        .frame(width: 45, alignment: .trailing)
                }
            }
        }
        .padding(16)
        .background(.ultraThinMaterial)
    }
    
    // MARK: - Gestures
    
    private var sliceNavigationGesture: some Gesture {
        DragGesture(minimumDistance: 20)
            .onEnded { value in
                let verticalMovement = value.translation.height
                
                if verticalMovement < -30 && currentSliceIndex < series.sliceCount - 1 {
                    withAnimation(.easeOut(duration: 0.1)) {
                        currentSliceIndex += 1
                    }
                } else if verticalMovement > 30 && currentSliceIndex > 0 {
                    withAnimation(.easeOut(duration: 0.1)) {
                        currentSliceIndex -= 1
                    }
                }
            }
    }
    
    // MARK: - Helpers
    
    private var currentMetadata: DICOMMetadata? {
        guard currentSliceIndex < series.files.count else { return nil }
        return series.files[currentSliceIndex].metadata
    }
    
    private func tryNextSlice() {
        if currentSliceIndex < series.sliceCount - 1 {
            currentSliceIndex += 1
        } else if currentSliceIndex > 0 {
            currentSliceIndex = 0
        }
    }
    
    private func loadCurrentSlice() async {
        guard series.sliceCount > 0 else {
            await MainActor.run {
                errorMessage = "No DICOM files in series"
                diagnosticInfo = nil
                isLoading = false
            }
            return
        }
        
        guard currentSliceIndex < series.files.count else {
            await MainActor.run {
                errorMessage = "Invalid slice index: \(currentSliceIndex + 1) of \(series.sliceCount)"
                diagnosticInfo = nil
                isLoading = false
            }
            return
        }
        
        let file = series.files[currentSliceIndex]
        let result = imageConverter.convertWithDiagnostics(
            file,
            windowCenter: windowCenter,
            windowWidth: windowWidth
        )
        
        await MainActor.run {
            diagnosticInfo = result.diagnosticInfo
            
            if let image = result.image {
                currentImage = Image(uiImage: image)
                errorMessage = nil
            } else {
                currentImage = nil
                errorMessage = result.error?.errorDescription ?? "Unknown error"
            }
            isLoading = false
        }
    }
}

// MARK: - Preview

//#if DEBUG
//#Preview("Mock CT Series") {
//    DICOMViewerView(
//        series: .mock(
//            sliceCount: 12,
//            pattern: .skull
//        )
//    )
//}

//#Preview("Gradient Pattern") {
//    DICOMViewerView(series: .mock(sliceCount: 5, pattern: .gradient))
//}

#Preview("Checkerboard") {
    DICOMViewerView(series: .mock(sliceCount: 3, pattern: .checkerboard))
}
//#endif
