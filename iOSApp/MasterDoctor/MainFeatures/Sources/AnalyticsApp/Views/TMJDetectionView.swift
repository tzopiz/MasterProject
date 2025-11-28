//  Created by AI Assistant on 24.11.2025.

import SwiftUI
import UniformTypeIdentifiers
import FoundationInternal
import CoreSwiftUI

public struct TMJDetectionView: View {
    @State private var selectedFiles: [URL] = []
    @State private var selectedFolderURL: URL?
    @State private var analysisResult: AnalysisResult?
    @State private var statusMessage: String = "Select DICOM folder to start"
    @State private var isProcessing = false
    @State private var showFilePicker = false
    @State private var showFileNames = false
    @State private var showDICOMViewer = false
    @State private var dicomSeries: DICOMSeries?
    @State private var isLoadingSeries = false

    @Environment(\.deps) private var deps

    public init() {}
    
    public var body: some View {
        NavigationStack {
            List {
                // DICOM Files Section
                Section {
                    if selectedFiles.isEmpty {
                        ContentUnavailableView {
                            Label("No DICOM Files", systemImage: "doc.viewfinder")
                        } description: {
                            Text("Select a folder containing DICOM files to get started")
                        }
                        .listRowBackground(Color.clear)
                    } else {
                        // Preview button
                        Button {
                            Task {
                                await loadAndShowDICOMViewer()
                            }
                        } label: {
                            HStack {
                                Image(systemName: "eye")
                                    .foregroundColor(.cyan)
                                Text("View DICOM Series")
                                Spacer()
                                if isLoadingSeries {
                                    ProgressView()
                                        .scaleEffect(0.8)
                                } else {
                                    Image(systemName: "chevron.right")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                        .disabled(isLoadingSeries)
                        
                        // File list toggle
                        if showFileNames {
                            ForEach(selectedFiles.prefix(20), id: \.self) { url in
                                HStack {
                                    Image(systemName: "doc.fill")
                                        .foregroundColor(.secondary)
                                        .font(.caption)
                                    Text(url.lastPathComponent)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            if selectedFiles.count > 20 {
                                Text("... and \(selectedFiles.count - 20) more files")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                } header: {
                    HStack {
                        Label("DICOM Files", systemImage: "folder")
                        Spacer()
                        if !selectedFiles.isEmpty {
                            Button {
                                withAnimation {
                                    showFileNames.toggle()
                                }
                            } label: {
                                Image(systemName: showFileNames ? "chevron.down" : "chevron.right")
                                    .contentTransition(.symbolEffect(.replace))
                                    .font(.caption)
                            }
                        }
                    }
                } footer: {
                    if selectedFiles.isEmpty {
                        Text("Tap + to select a DICOM folder")
                    } else {
                        Text("\(selectedFiles.count) DICOM files ready")
                    }
                }

                // Status Section
                Section {
                    HStack {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 8, height: 8)
                        Text(statusMessage)
                            .foregroundColor(isProcessing ? .orange : .primary)
                    }

                    if isProcessing {
                        ProgressView()
                            .progressViewStyle(.linear)
                    }
                } header: {
                    Label("Status", systemImage: "waveform.path.ecg")
                }

                // Analysis Results
                analysisResult?.makeView()

                // Action Button
                if selectedFiles.nilIfEmpty != nil && !isProcessing {
                    Section {
                        Button {
                            Task {
                                // TODO: Implement analysis
                            }
                        } label: {
                            Label("Start TMJ Detection", systemImage: "brain")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .listRowBackground(Color.clear)
                        .listRowInsets(EdgeInsets())
                    }
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showFilePicker = true
                    } label: {
                        Image(systemName: "plus")
                    }
                }
            }
            .navigationTitle("TMJ Detection")
            .fileImporter(
                isPresented: $showFilePicker,
                statusMessage: $statusMessage,
                selectedFiles: $selectedFiles,
                selectedFolderURL: $selectedFolderURL
            )
            .fullScreenCover(isPresented: $showDICOMViewer) {
                if let series = dicomSeries {
                    DICOMViewerScreen(series: series)
                }
            }
        }
    }
    
    // MARK: - Computed Properties
    
    private var statusColor: Color {
        if isProcessing {
            .orange
        } else if selectedFiles.isEmpty {
            .gray
        } else if analysisResult != nil {
            .green
        } else {
            .blue
        }
    }
    
    // MARK: - Methods
    
    private func loadAndShowDICOMViewer() async {
        guard let folderURL = selectedFolderURL else {
            statusMessage = "No folder selected"
            return
        }
        
        isLoadingSeries = true
        
        // Start accessing security-scoped resource
        guard folderURL.startAccessingSecurityScopedResource() else {
            statusMessage = "Cannot access folder"
            isLoadingSeries = false
            return
        }
        
        defer {
            folderURL.stopAccessingSecurityScopedResource()
        }
        
        do {
            let loader = DICOMSeriesLoader()
            let series = try await loader.loadFiles(from: selectedFiles)
            
            await MainActor.run {
                dicomSeries = series
                isLoadingSeries = false
                
                if series.sliceCount > 0 {
                    showDICOMViewer = true
                } else {
                    statusMessage = "No valid DICOM files found"
                }
            }
        } catch {
            await MainActor.run {
                statusMessage = "Error loading DICOM: \(error.localizedDescription)"
                isLoadingSeries = false
            }
        }
    }
}

// MARK: - File Importer Extension

extension View {
    fileprivate func fileImporter(
        isPresented: Binding<Bool>,
        statusMessage: Binding<String>,
        selectedFiles: Binding<[URL]>,
        selectedFolderURL: Binding<URL?>
    ) -> some View {
        self.fileImporter(
            isPresented: isPresented,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let folderURL = urls.first else { return }

                guard folderURL.startAccessingSecurityScopedResource() else {
                    statusMessage.wrappedValue = "Cannot access folder"
                    return
                }

                defer { folderURL.stopAccessingSecurityScopedResource() }

                let fileManager = FileManager.default
                do {
                    let contents = try fileManager.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil)
                    let dcmFiles = contents.filter { $0.pathExtension.lowercased() == "dcm" }
                    
                    selectedFiles.wrappedValue = dcmFiles
                    selectedFolderURL.wrappedValue = folderURL

                    if dcmFiles.isEmpty {
                        statusMessage.wrappedValue = "No DICOM files found in folder"
                    } else {
                        statusMessage.wrappedValue = "Ready to process \(dcmFiles.count) files"
                    }
                } catch {
                    statusMessage.wrappedValue = "Error reading folder: \(error.localizedDescription)"
                }

            case .failure(let error):
                statusMessage.wrappedValue = "Error: \(error.localizedDescription)"
            }
        }
    }
}

#Preview {
    TMJDetectionView()
}
