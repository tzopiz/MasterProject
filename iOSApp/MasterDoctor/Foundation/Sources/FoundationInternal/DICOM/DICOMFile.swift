//  Created by Dmitrii Korchagin on 26.11.2025.

import Foundation
import CoreGraphics

/// Represents a parsed DICOM file with metadata and pixel data
public struct DICOMFile: Sendable {
    public let url: URL
    public let metadata: DICOMMetadata
    public let pixelData: Data?
    public let parseInfo: DICOMParseInfo
    
    public init(url: URL, metadata: DICOMMetadata, pixelData: Data?, parseInfo: DICOMParseInfo = .init()) {
        self.url = url
        self.metadata = metadata
        self.pixelData = pixelData
        self.parseInfo = parseInfo
    }
}

/// Information about DICOM parsing process
public struct DICOMParseInfo: Sendable {
    public let hasDICMPrefix: Bool
    public let foundTags: Set<String>
    public let transferSyntax: String?
    public let isCompressed: Bool
    public let warnings: [String]
    
    public init(
        hasDICMPrefix: Bool = false,
        foundTags: Set<String> = [],
        transferSyntax: String? = nil,
        isCompressed: Bool = false,
        warnings: [String] = [],
    ) {
        self.hasDICMPrefix = hasDICMPrefix
        self.foundTags = foundTags
        self.transferSyntax = transferSyntax
        self.isCompressed = isCompressed
        self.warnings = warnings
    }
}

/// DICOM file metadata
public struct DICOMMetadata: Sendable {
    // Image dimensions
    public let rows: Int
    public let columns: Int
    public let bitsAllocated: Int
    public let bitsStored: Int
    public let highBit: Int
    public let pixelRepresentation: Int // 0 = unsigned, 1 = signed
    public let samplesPerPixel: Int
    public let photometricInterpretation: String
    
    // Whether values came from file or are defaults
    public let hasExplicitDimensions: Bool
    
    // Window/Level settings
    public let windowCenter: Double?
    public let windowWidth: Double?
    public let rescaleSlope: Double
    public let rescaleIntercept: Double
    
    // Spatial information
    public let pixelSpacing: (row: Double, column: Double)?
    public let sliceThickness: Double?
    public let sliceLocation: Double?
    public let imagePosition: (x: Double, y: Double, z: Double)?
    public let instanceNumber: Int?
    
    // Patient/Study information
    public let patientName: String?
    public let patientID: String?
    public let studyDate: String?
    public let seriesDescription: String?
    public let modality: String?
    
    public init(
        rows: Int,
        columns: Int,
        bitsAllocated: Int,
        bitsStored: Int,
        highBit: Int,
        pixelRepresentation: Int,
        samplesPerPixel: Int,
        photometricInterpretation: String,
        hasExplicitDimensions: Bool = true,
        windowCenter: Double?,
        windowWidth: Double?,
        rescaleSlope: Double,
        rescaleIntercept: Double,
        pixelSpacing: (row: Double, column: Double)?,
        sliceThickness: Double?,
        sliceLocation: Double?,
        imagePosition: (x: Double, y: Double, z: Double)?,
        instanceNumber: Int?,
        patientName: String?,
        patientID: String?,
        studyDate: String?,
        seriesDescription: String?,
        modality: String?,
    ) {
        self.rows = rows
        self.columns = columns
        self.bitsAllocated = bitsAllocated
        self.bitsStored = bitsStored
        self.highBit = highBit
        self.pixelRepresentation = pixelRepresentation
        self.samplesPerPixel = samplesPerPixel
        self.photometricInterpretation = photometricInterpretation
        self.hasExplicitDimensions = hasExplicitDimensions
        self.windowCenter = windowCenter
        self.windowWidth = windowWidth
        self.rescaleSlope = rescaleSlope
        self.rescaleIntercept = rescaleIntercept
        self.pixelSpacing = pixelSpacing
        self.sliceThickness = sliceThickness
        self.sliceLocation = sliceLocation
        self.imagePosition = imagePosition
        self.instanceNumber = instanceNumber
        self.patientName = patientName
        self.patientID = patientID
        self.studyDate = studyDate
        self.seriesDescription = seriesDescription
        self.modality = modality
    }
}

// MARK: - Default Metadata

extension DICOMMetadata {
    public static func makeDefault(rows: Int = 512, columns: Int = 512) -> DICOMMetadata {
        DICOMMetadata(
            rows: rows,
            columns: columns,
            bitsAllocated: 16,
            bitsStored: 12,
            highBit: 11,
            pixelRepresentation: 0,
            samplesPerPixel: 1,
            photometricInterpretation: "MONOCHROME2",
            hasExplicitDimensions: false,
            windowCenter: nil,
            windowWidth: nil,
            rescaleSlope: 1.0,
            rescaleIntercept: 0.0,
            pixelSpacing: nil,
            sliceThickness: nil,
            sliceLocation: nil,
            imagePosition: nil,
            instanceNumber: nil,
            patientName: nil,
            patientID: nil,
            studyDate: nil,
            seriesDescription: nil,
            modality: nil,
        )
    }
}
