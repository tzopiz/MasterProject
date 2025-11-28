//  Created by Dmitrii Korchagin on 26.11.2025.

import Foundation

/// Represents a DICOM tag with group and element numbers
public struct DICOMTag: Hashable, Sendable {
    public let group: UInt16
    public let element: UInt16
    
    public init(group: UInt16, element: UInt16) {
        self.group = group
        self.element = element
    }
}

// MARK: - Common DICOM Tags

public extension DICOMTag {
    // Patient Information
    static let patientName = DICOMTag(group: 0x0010, element: 0x0010)
    static let patientID = DICOMTag(group: 0x0010, element: 0x0020)
    static let patientBirthDate = DICOMTag(group: 0x0010, element: 0x0030)
    static let patientSex = DICOMTag(group: 0x0010, element: 0x0040)
    
    // Study Information
    static let studyDate = DICOMTag(group: 0x0008, element: 0x0020)
    static let studyTime = DICOMTag(group: 0x0008, element: 0x0030)
    static let studyDescription = DICOMTag(group: 0x0008, element: 0x1030)
    static let studyInstanceUID = DICOMTag(group: 0x0020, element: 0x000D)
    
    // Series Information
    static let seriesInstanceUID = DICOMTag(group: 0x0020, element: 0x000E)
    static let seriesNumber = DICOMTag(group: 0x0020, element: 0x0011)
    static let seriesDescription = DICOMTag(group: 0x0008, element: 0x103E)
    static let modality = DICOMTag(group: 0x0008, element: 0x0060)
    
    // Image Information
    static let instanceNumber = DICOMTag(group: 0x0020, element: 0x0013)
    static let sopInstanceUID = DICOMTag(group: 0x0008, element: 0x0018)
    static let imagePosition = DICOMTag(group: 0x0020, element: 0x0032)
    static let imageOrientation = DICOMTag(group: 0x0020, element: 0x0037)
    static let sliceThickness = DICOMTag(group: 0x0018, element: 0x0050)
    static let sliceLocation = DICOMTag(group: 0x0020, element: 0x1041)
    
    // Pixel Data Information
    static let rows = DICOMTag(group: 0x0028, element: 0x0010)
    static let columns = DICOMTag(group: 0x0028, element: 0x0011)
    static let bitsAllocated = DICOMTag(group: 0x0028, element: 0x0100)
    static let bitsStored = DICOMTag(group: 0x0028, element: 0x0101)
    static let highBit = DICOMTag(group: 0x0028, element: 0x0102)
    static let pixelRepresentation = DICOMTag(group: 0x0028, element: 0x0103)
    static let samplesPerPixel = DICOMTag(group: 0x0028, element: 0x0002)
    static let photometricInterpretation = DICOMTag(group: 0x0028, element: 0x0004)
    static let pixelSpacing = DICOMTag(group: 0x0028, element: 0x0030)
    
    // Window Settings
    static let windowCenter = DICOMTag(group: 0x0028, element: 0x1050)
    static let windowWidth = DICOMTag(group: 0x0028, element: 0x1051)
    static let rescaleIntercept = DICOMTag(group: 0x0028, element: 0x1052)
    static let rescaleSlope = DICOMTag(group: 0x0028, element: 0x1053)
    
    // Pixel Data
    static let pixelData = DICOMTag(group: 0x7FE0, element: 0x0010)
    
    // Transfer Syntax
    static let transferSyntaxUID = DICOMTag(group: 0x0002, element: 0x0010)
}

extension DICOMTag: CustomStringConvertible {
    public var description: String {
        String(format: "(%04X,%04X)", group, element)
    }
}

