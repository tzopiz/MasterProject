//  Created by Dmitrii Korchagin on 26.11.2025.

import Foundation

/// Errors that can occur during DICOM parsing
public enum DICOMParserError: Error, LocalizedError {
    case fileNotFound
    case invalidDICOMFile
    case missingRequiredTag(DICOMTag)
    case unsupportedTransferSyntax(String)
    case invalidPixelData
    case readError(String)
    
    public var errorDescription: String? {
        switch self {
        case .fileNotFound: "DICOM file not found"
        case .invalidDICOMFile: "Invalid DICOM file format"
        case .missingRequiredTag(let tag): "Missing required DICOM tag: \(tag)"
        case .unsupportedTransferSyntax(let syntax): "Unsupported transfer syntax: \(syntax)"
        case .invalidPixelData: "Invalid or missing pixel data"
        case .readError(let message): "Read error: \(message)"
        }
    }
}

/// Known compressed transfer syntaxes
private let compressedTransferSyntaxes: Set<String> = [
    "1.2.840.10008.1.2.4.50",  // JPEG Baseline
    "1.2.840.10008.1.2.4.51",  // JPEG Extended
    "1.2.840.10008.1.2.4.57",  // JPEG Lossless
    "1.2.840.10008.1.2.4.70",  // JPEG Lossless First-order
    "1.2.840.10008.1.2.4.80",  // JPEG-LS Lossless
    "1.2.840.10008.1.2.4.81",  // JPEG-LS Lossy
    "1.2.840.10008.1.2.4.90",  // JPEG 2000 Lossless
    "1.2.840.10008.1.2.4.91",  // JPEG 2000
    "1.2.840.10008.1.2.5",     // RLE Lossless
]

/// Parser for DICOM files
public struct DICOMParser: Sendable {

    public init() {}

    /// Parse a DICOM file from URL
    public func parse(url: URL) throws -> DICOMFile {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DICOMParserError.fileNotFound
        }
        
        let data = try Data(contentsOf: url)
        return try parse(data: data, url: url)
    }
    
    /// Parse DICOM data
    public func parse(data: Data, url: URL) throws -> DICOMFile {
        var reader = DICOMDataReader(data: data)
        var warnings: [String] = []
        var foundTagNames: Set<String> = []
        
        // Check for DICOM preamble and magic bytes
        let hasDICMPrefix = checkDICOMPrefix(reader: &reader)
        
        if !hasDICMPrefix {
            // Try parsing without prefix (some DICOM files don't have it)
            reader = DICOMDataReader(data: data)
            warnings.append("No DICM prefix found, attempting raw parse")
        }
        
        var elements: [DICOMTag: Any] = [:]
        var pixelData: Data?
        var isExplicitVR = true
        var isLittleEndian = true
        var transferSyntax: String?
        
        while reader.hasMoreData {
            do {
                let (tag, value, vr) = try parseDataElement(
                    reader: &reader,
                    isExplicitVR: isExplicitVR,
                    isLittleEndian: isLittleEndian
                )
                
                // Track found tags
                foundTagNames.insert(tag.description)
                
                // Check for transfer syntax in meta header
                if tag == .transferSyntaxUID, let tsStr = value as? String {
                    transferSyntax = tsStr.trimmingCharacters(in: CharacterSet(charactersIn: " \0"))
                    (isExplicitVR, isLittleEndian) = parseTransferSyntax(tsStr)
                }
                
                // Handle pixel data specially
                if tag == .pixelData {
                    pixelData = value as? Data
                } else {
                    elements[tag] = value
                }
                
            } catch DICOMParserError.readError {
                // End of data or parse error - stop parsing
                break
            }
        }
        
        // Check if compressed
        let isCompressed = transferSyntax.map { compressedTransferSyntaxes.contains($0) } ?? false
        if isCompressed {
            warnings.append("Compressed transfer syntax detected: \(transferSyntax ?? "unknown"). Decompression not supported.")
        }
        
        // Build metadata and check for missing important tags
        let (metadata, metadataWarnings) = buildMetadataWithWarnings(from: elements)
        warnings.append(contentsOf: metadataWarnings)
        
        let parseInfo = DICOMParseInfo(
            hasDICMPrefix: hasDICMPrefix,
            foundTags: foundTagNames,
            transferSyntax: transferSyntax,
            isCompressed: isCompressed,
            warnings: warnings
        )
        
        return DICOMFile(url: url, metadata: metadata, pixelData: pixelData, parseInfo: parseInfo)
    }
    
    // MARK: - Private Methods
    
    private func checkDICOMPrefix(reader: inout DICOMDataReader) -> Bool {
        // Skip 128-byte preamble
        guard reader.canRead(128) else { return false }
        reader.skip(128)
        
        // Check for "DICM" magic bytes
        guard reader.canRead(4) else { return false }
        let magic = reader.readBytes(4)
        
        return magic == Data([0x44, 0x49, 0x43, 0x4D]) // "DICM"
    }
    
    private func parseDataElement(
        reader: inout DICOMDataReader,
        isExplicitVR: Bool,
        isLittleEndian: Bool
    ) throws -> (tag: DICOMTag, value: Any, vr: String?) {
        guard reader.canRead(4) else {
            throw DICOMParserError.readError("Not enough data for tag")
        }
        
        let group = reader.readUInt16(littleEndian: isLittleEndian)
        let element = reader.readUInt16(littleEndian: isLittleEndian)
        let tag = DICOMTag(group: group, element: element)
        
        // Meta header (group 0x0002) is always explicit VR little endian
        let useExplicitVR = (group == 0x0002) || isExplicitVR
        
        var vr: String?
        var length: UInt32
        
        if useExplicitVR {
            guard reader.canRead(2) else {
                throw DICOMParserError.readError("Not enough data for VR")
            }
            vr = reader.readString(2)
            
            // VRs that use 4-byte length field
            let longVRs = ["OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UN", "UR", "UT"]
            
            if longVRs.contains(vr ?? "") {
                reader.skip(2) // Reserved bytes
                guard reader.canRead(4) else {
                    throw DICOMParserError.readError("Not enough data for length")
                }
                length = reader.readUInt32(littleEndian: isLittleEndian)
            } else {
                guard reader.canRead(2) else {
                    throw DICOMParserError.readError("Not enough data for length")
                }
                length = UInt32(reader.readUInt16(littleEndian: isLittleEndian))
            }
        } else {
            // Implicit VR
            guard reader.canRead(4) else {
                throw DICOMParserError.readError("Not enough data for length")
            }
            length = reader.readUInt32(littleEndian: isLittleEndian)
        }
        
        // Undefined length
        if length == 0xFFFFFFFF {
            // For sequences or encapsulated pixel data
            let value = try parseUndefinedLengthData(reader: &reader, isLittleEndian: isLittleEndian)
            return (tag, value, vr)
        }
        
        guard reader.canRead(Int(length)) else {
            throw DICOMParserError.readError("Not enough data for value")
        }
        
        let valueData = reader.readBytes(Int(length))
        let value = convertValue(data: valueData, vr: vr, isLittleEndian: isLittleEndian)
        
        return (tag, value, vr)
    }
    
    private func parseUndefinedLengthData(
        reader: inout DICOMDataReader,
        isLittleEndian: Bool
    ) throws -> Data {
        var data = Data()
        
        // Look for sequence delimitation item (FFFE,E0DD)
        while reader.hasMoreData {
            guard reader.canRead(8) else { break }
            
            let group = reader.readUInt16(littleEndian: isLittleEndian)
            let element = reader.readUInt16(littleEndian: isLittleEndian)
            let length = reader.readUInt32(littleEndian: isLittleEndian)
            
            if group == 0xFFFE && element == 0xE0DD {
                // Sequence delimitation
                break
            }
            
            if length != 0xFFFFFFFF && reader.canRead(Int(length)) {
                data.append(reader.readBytes(Int(length)))
            }
        }
        
        return data
    }
    
    private func convertValue(data: Data, vr: String?, isLittleEndian: Bool) -> Any {
        guard let vr = vr else {
            return data
        }
        
        switch vr {
        case "US": // Unsigned Short
            guard data.count >= 2 else { return data }
            return data.withUnsafeBytes { ptr -> UInt16 in
                let value = ptr.load(as: UInt16.self)
                return isLittleEndian ? value : value.byteSwapped
            }
            
        case "SS": // Signed Short
            guard data.count >= 2 else { return data }
            return data.withUnsafeBytes { ptr -> Int16 in
                let value = ptr.load(as: Int16.self)
                return isLittleEndian ? value : value.byteSwapped
            }
            
        case "UL": // Unsigned Long
            guard data.count >= 4 else { return data }
            return data.withUnsafeBytes { ptr -> UInt32 in
                let value = ptr.load(as: UInt32.self)
                return isLittleEndian ? value : value.byteSwapped
            }
            
        case "SL": // Signed Long
            guard data.count >= 4 else { return data }
            return data.withUnsafeBytes { ptr -> Int32 in
                let value = ptr.load(as: Int32.self)
                return isLittleEndian ? value : value.byteSwapped
            }
            
        case "FL": // Float
            guard data.count >= 4 else { return data }
            return data.withUnsafeBytes { ptr -> Float in
                ptr.load(as: Float.self)
            }
            
        case "FD": // Double
            guard data.count >= 8 else { return data }
            return data.withUnsafeBytes { ptr -> Double in
                ptr.load(as: Double.self)
            }
            
        case "DS": // Decimal String
            let str = String(data: data, encoding: .ascii)?.trimmingCharacters(in: .whitespaces) ?? ""
            return Double(str) ?? str
            
        case "IS": // Integer String
            let str = String(data: data, encoding: .ascii)?.trimmingCharacters(in: .whitespaces) ?? ""
            return Int(str) ?? str
            
        case "LO", "SH", "PN", "CS", "UI", "AE", "AS", "DA", "TM", "LT", "ST", "UT":
            // String types
            return String(data: data, encoding: .ascii)?
                .trimmingCharacters(in: CharacterSet(charactersIn: " \0")) ?? ""
            
        case "OB", "OW", "OD", "OF", "OL", "UN":
            // Binary data
            return data
            
        default:
            return data
        }
    }
    
    private func parseTransferSyntax(_ uid: String) -> (explicitVR: Bool, littleEndian: Bool) {
        switch uid.trimmingCharacters(in: CharacterSet(charactersIn: " \0")) {
        case "1.2.840.10008.1.2":      // Implicit VR Little Endian
            (false, true)
        case "1.2.840.10008.1.2.1":    // Explicit VR Little Endian
            (true, true)
        case "1.2.840.10008.1.2.2":    // Explicit VR Big Endian
            (true, false)
        default:
            // Default to Explicit VR Little Endian for compressed syntaxes
            (true, true)
        }
    }
    
    private func buildMetadataWithWarnings(from elements: [DICOMTag: Any]) -> (DICOMMetadata, [String]) {
        var warnings: [String] = []
        
        let rowsValue = elements[.rows] as? UInt16
        let columnsValue = elements[.columns] as? UInt16
        
        let hasExplicitDimensions = rowsValue != nil && columnsValue != nil
        
        if rowsValue == nil {
            warnings.append("Missing Rows tag (0028,0010)")
        }
        if columnsValue == nil {
            warnings.append("Missing Columns tag (0028,0011)")
        }
        
        let rows = rowsValue.map(Int.init) ?? 0
        let columns = columnsValue.map(Int.init) ?? 0
        
        let bitsAllocatedValue = elements[.bitsAllocated] as? UInt16
        if bitsAllocatedValue == nil {
            warnings.append("Missing BitsAllocated tag (0028,0100)")
        }
        let bitsAllocated = bitsAllocatedValue.map(Int.init) ?? 16
        
        let bitsStored = (elements[.bitsStored] as? UInt16).map(Int.init) ?? 12
        let highBit = (elements[.highBit] as? UInt16).map(Int.init) ?? 11
        let pixelRepresentation = (elements[.pixelRepresentation] as? UInt16).map(Int.init) ?? 0
        let samplesPerPixel = (elements[.samplesPerPixel] as? UInt16).map(Int.init) ?? 1
        let photometricInterpretation = (elements[.photometricInterpretation] as? String) ?? "MONOCHROME2"
        
        // Window settings
        let windowCenter = elements[.windowCenter] as? Double
        let windowWidth = elements[.windowWidth] as? Double
        let rescaleSlope = (elements[.rescaleSlope] as? Double) ?? 1.0
        let rescaleIntercept = (elements[.rescaleIntercept] as? Double) ?? 0.0
        
        // Spatial information
        var pixelSpacing: (row: Double, column: Double)?
        if let psStr = elements[.pixelSpacing] as? String {
            let parts = psStr.split(separator: "\\")
            if parts.count >= 2,
               let row = Double(parts[0]),
               let col = Double(parts[1]) {
                pixelSpacing = (row, col)
            }
        }
        
        let sliceThickness = elements[.sliceThickness] as? Double
        let sliceLocation = elements[.sliceLocation] as? Double
        
        var imagePosition: (x: Double, y: Double, z: Double)?
        if let posStr = elements[.imagePosition] as? String {
            let parts = posStr.split(separator: "\\")
            if parts.count >= 3,
               let x = Double(parts[0]),
               let y = Double(parts[1]),
               let z = Double(parts[2]) {
                imagePosition = (x, y, z)
            }
        }
        
        let instanceNumber = elements[.instanceNumber] as? Int
        
        // Patient/Study info
        let patientName = elements[.patientName] as? String
        let patientID = elements[.patientID] as? String
        let studyDate = elements[.studyDate] as? String
        let seriesDescription = elements[.seriesDescription] as? String
        let modality = elements[.modality] as? String
        
        let metadata = DICOMMetadata(
            rows: rows,
            columns: columns,
            bitsAllocated: bitsAllocated,
            bitsStored: bitsStored,
            highBit: highBit,
            pixelRepresentation: pixelRepresentation,
            samplesPerPixel: samplesPerPixel,
            photometricInterpretation: photometricInterpretation,
            hasExplicitDimensions: hasExplicitDimensions,
            windowCenter: windowCenter,
            windowWidth: windowWidth,
            rescaleSlope: rescaleSlope,
            rescaleIntercept: rescaleIntercept,
            pixelSpacing: pixelSpacing,
            sliceThickness: sliceThickness,
            sliceLocation: sliceLocation,
            imagePosition: imagePosition,
            instanceNumber: instanceNumber,
            patientName: patientName,
            patientID: patientID,
            studyDate: studyDate,
            seriesDescription: seriesDescription,
            modality: modality
        )
        
        return (metadata, warnings)
    }
}

// MARK: - Data Reader Helper

fileprivate struct DICOMDataReader {
    private let data: Data
    private var offset: Int = 0
    
    init(data: Data) {
        self.data = data
    }
    
    var hasMoreData: Bool {
        offset < data.count
    }
    
    func canRead(_ count: Int) -> Bool {
        offset + count <= data.count
    }
    
    mutating func skip(_ count: Int) {
        offset += count
    }
    
    mutating func readBytes(_ count: Int) -> Data {
        let result = data.subdata(in: offset..<min(offset + count, data.count))
        offset += count
        return result
    }
    
    mutating func readUInt16(littleEndian: Bool) -> UInt16 {
        let bytes = readBytes(2)
        guard bytes.count >= 2 else { return 0 }
        
        return bytes.withUnsafeBytes { ptr in
            let value = ptr.load(as: UInt16.self)
            return littleEndian ? value : value.byteSwapped
        }
    }
    
    mutating func readUInt32(littleEndian: Bool) -> UInt32 {
        let bytes = readBytes(4)
        guard bytes.count >= 4 else { return 0 }
        
        return bytes.withUnsafeBytes { ptr in
            let value = ptr.load(as: UInt32.self)
            return littleEndian ? value : value.byteSwapped
        }
    }
    
    mutating func readString(_ count: Int) -> String {
        let bytes = readBytes(count)
        return String(data: bytes, encoding: .ascii) ?? ""
    }
}
