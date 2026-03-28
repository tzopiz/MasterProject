//  Created by Dmitrii Korchagin on 26.11.2025.

import Foundation

/// Represents a series of DICOM files (e.g., a CT scan volume)
public struct DICOMSeries: Sendable {
    public let files: [DICOMFile]
    
    /// Patient name from the first file
    public var patientName: String? {
        files.first?.metadata.patientName
    }
    
    /// Study date from the first file
    public var studyDate: String? {
        files.first?.metadata.studyDate
    }
    
    /// Series description from the first file
    public var seriesDescription: String? {
        files.first?.metadata.seriesDescription
    }
    
    /// Modality from the first file
    public var modality: String? {
        files.first?.metadata.modality
    }
    
    /// Number of slices in the series
    public var sliceCount: Int {
        files.count
    }
    
    public init(files: [DICOMFile]) {
        // Sort by slice location or instance number
        self.files = files.sorted { f1, f2 in
            if let loc1 = f1.metadata.sliceLocation,
               let loc2 = f2.metadata.sliceLocation {
                return loc1 < loc2
            }
            if let num1 = f1.metadata.instanceNumber,
               let num2 = f2.metadata.instanceNumber {
                return num1 < num2
            }
            return f1.url.lastPathComponent < f2.url.lastPathComponent
        }
    }
}

// MARK: - Series Loader

/// Loads a series of DICOM files from a directory
public actor DICOMSeriesLoader {
    private let parser = DICOMParser()
    
    public init() {}
    
    /// Load all DICOM files from a directory
    public func loadSeries(from directoryURL: URL) async throws -> DICOMSeries {
        let fileManager = FileManager.default
        
        let contents = try fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey]
        )
        
        let dicomURLs = contents.filter { url in
            url.pathExtension.lowercased() == "dcm" || 
            url.pathExtension.isEmpty // Some DICOM files have no extension
        }
        
        var files: [DICOMFile] = []
        
        for url in dicomURLs {
            do {
                let file = try parser.parse(url: url)
                files.append(file)
            } catch {
                // Skip files that can't be parsed
                continue
            }
        }
        
        return DICOMSeries(files: files)
    }
    
    /// Load DICOM files from specific URLs
    public func loadFiles(from urls: [URL]) async throws -> DICOMSeries {
        var files: [DICOMFile] = []
        
        for url in urls {
            do {
                let file = try parser.parse(url: url)
                files.append(file)
            } catch {
                continue
            }
        }
        
        return DICOMSeries(files: files)
    }
}

