//  Created by AI Assistant on 24.11.2025.

import SwiftUI
import UniformTypeIdentifiers
import CoreNetwork
import FoundationInternal
import CoreSwiftUI

public struct TMJDetectionView: View {
    @State private var selectedFiles: [URL] = []
    @State private var analysisResult: AnalysisResult?
    @State private var statusMessage: String = "Select DICOM folder to start"
    @State private var isProcessing = false
    @State private var showFilePicker = false
    @State private var showFileNames = false

    private let fetchService: TMJDetectionFetchService

    @Environment(\.deps) private var deps

    public init() {
        
    }
    
    public var body: some View {
        NavigationStack {
            List {
                Section {
                    if selectedFiles.isEmpty {
                        Text("Here will be your files")
                            .foregroundStyle(.placeholder)
                    } else {
                        if showFileNames {
                            ForEach(selectedFiles, id: \.self) { url in
                                Text(url.pathComponents.last ?? "")
                            }
                        }
                    }
                } header: {
                    HStack {
                        Label("Status", systemImage: "folder")
                        Spacer()
                        Button {
                            withAnimation {
                                showFileNames.toggle()
                            }
                        } label: {
                            Image(systemName: showFileNames ? "chevron.down" : "chevron.up")
                                .contentTransition(.symbolEffect(.replace))
                        }
                        .disabled(selectedFiles.isEmpty)
                    }
                } footer: {
                    if selectedFiles.isEmpty {
                        Text("No files selected")
                    } else {
                        Text("\(selectedFiles.count) DICOM files")
                    }
                }

                Section {
                    Text(statusMessage)
                        .foregroundColor(isProcessing ? .orange : .primary)

                    if isProcessing {
                        ProgressView()
                            .progressViewStyle(.linear)
                    }
                } header: {
                    Label("Status", systemImage: "apple.intelligence")
                }

                analysisResult?.makeView()

                if let selectedFiles = selectedFiles.nilIfEmpty, !isProcessing {
                    Button {
                        Task {
//                            await runAnalysis()
                        }
                    } label: {
                        Label("Start TMJ Detection", systemImage: "brain")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
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
            )
        }
    }
}

extension View {
    fileprivate func fileImporter(
        isPresented: Binding<Bool>,
        statusMessage: Binding<String>,
        selectedFiles: Binding<[URL]>,
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
                    selectedFiles.wrappedValue = contents.filter { $0.pathExtension.lowercased() == "dcm" }

                    if selectedFiles.isEmpty {
                        statusMessage.wrappedValue = "No DICOM files found in folder"
                    } else {
                        statusMessage.wrappedValue = "Ready to process \(selectedFiles.count) files"
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
