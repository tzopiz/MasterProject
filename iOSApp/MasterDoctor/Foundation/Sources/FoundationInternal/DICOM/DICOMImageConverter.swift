//  Created by Dmitrii Korchagin on 26.11.2025.

import Foundation
import CoreGraphics
import ImageIO

import UIKit
public typealias PlatformImage = UIImage

/// Errors that can occur during DICOM image conversion
public enum DICOMImageError: Error, LocalizedError, Sendable {
    case noPixelData
    case insufficientPixelData(expected: Int, actual: Int)
    case unsupportedBitsAllocated(Int)
    case failedToCreateDataProvider
    case failedToCreateCGImage
    case invalidDimensions(width: Int, height: Int)
    case parsingFailed(String)
    case compressedNotSupported(String)
    case missingDimensions
    
    public var errorDescription: String? {
        switch self {
        case .noPixelData:
            return "No pixel data found in DICOM file"
        case .insufficientPixelData(let expected, let actual):
            return "Insufficient pixel data: expected \(expected) bytes, got \(actual) bytes"
        case .unsupportedBitsAllocated(let bits):
            return "Unsupported bits allocated: \(bits) (only 8 and 16 are supported)"
        case .failedToCreateDataProvider:
            return "Failed to create image data provider"
        case .failedToCreateCGImage:
            return "Failed to create CGImage"
        case .invalidDimensions(let width, let height):
            return "Invalid image dimensions: \(width)×\(height)"
        case .parsingFailed(let reason):
            return "DICOM parsing failed: \(reason)"
        case .compressedNotSupported(let syntax):
            return "Compressed DICOM not supported: \(syntax)"
        case .missingDimensions:
            return "Image dimensions not found in DICOM file"
        }
    }
}

/// Result of DICOM image conversion with diagnostic info
public struct DICOMConversionResult: Sendable {
    public let image: PlatformImage?
    public let error: DICOMImageError?
    public let diagnosticInfo: DICOMDiagnosticInfo
    
    public var isSuccess: Bool { image != nil }
}

/// Diagnostic information about DICOM file
public struct DICOMDiagnosticInfo: Sendable {
    public let dimensions: String
    public let bitsAllocated: Int
    public let bitsStored: Int
    public let pixelRepresentation: String
    public let photometricInterpretation: String
    public let pixelDataSize: Int
    public let expectedDataSize: Int
    public let hasWindowSettings: Bool
    public let modality: String?
    public let hasExplicitDimensions: Bool
    public let hasDICMPrefix: Bool
    public let isCompressed: Bool
    public let transferSyntax: String?
    public let warnings: [String]
    
    public var summary: String {
        """
        Dimensions: \(dimensions)\(hasExplicitDimensions ? "" : " (default)")
        Bits: \(bitsAllocated) allocated, \(bitsStored) stored
        Pixel Rep: \(pixelRepresentation)
        Photometric: \(photometricInterpretation)
        Modality: \(modality ?? "Unknown")
        Pixel Data: \(pixelDataSize) bytes (expected: \(expectedDataSize))
        Window Settings: \(hasWindowSettings ? "Yes" : "No")
        DICM Prefix: \(hasDICMPrefix ? "Yes" : "No")
        Compressed: \(isCompressed ? "Yes" : "No")
        """
    }
}

/// Converts DICOM pixel data to displayable images
public struct DICOMImageConverter: Sendable {
    
    private let decompressor = DICOMDecompressor()

    public init() {}

    /// Convert a DICOM file to a platform image with detailed result
    public func convertWithDiagnostics(
        _ dicomFile: DICOMFile,
        windowCenter: Double? = nil,
        windowWidth: Double? = nil
    ) -> DICOMConversionResult {
        let metadata = dicomFile.metadata
        let parseInfo = dicomFile.parseInfo
        let pixelDataSize = dicomFile.pixelData?.count ?? 0
        let expectedSize = metadata.columns * metadata.rows * metadata.samplesPerPixel * (metadata.bitsAllocated / 8)
        
        let diagnosticInfo = DICOMDiagnosticInfo(
            dimensions: "\(metadata.columns)×\(metadata.rows)",
            bitsAllocated: metadata.bitsAllocated,
            bitsStored: metadata.bitsStored,
            pixelRepresentation: metadata.pixelRepresentation == 0 ? "Unsigned" : "Signed",
            photometricInterpretation: metadata.photometricInterpretation,
            pixelDataSize: pixelDataSize,
            expectedDataSize: expectedSize,
            hasWindowSettings: metadata.windowCenter != nil && metadata.windowWidth != nil,
            modality: metadata.modality,
            hasExplicitDimensions: metadata.hasExplicitDimensions,
            hasDICMPrefix: parseInfo.hasDICMPrefix,
            isCompressed: parseInfo.isCompressed,
            transferSyntax: parseInfo.transferSyntax,
            warnings: parseInfo.warnings
        )
        
        // Check if we have explicit dimensions
        guard metadata.hasExplicitDimensions else {
            return DICOMConversionResult(
                image: nil,
                error: .missingDimensions,
                diagnosticInfo: diagnosticInfo
            )
        }
        
        // Validate dimensions
        guard metadata.columns > 0 && metadata.rows > 0 else {
            return DICOMConversionResult(
                image: nil,
                error: .invalidDimensions(width: metadata.columns, height: metadata.rows),
                diagnosticInfo: diagnosticInfo
            )
        }
        
        // Check pixel data
        guard let originalPixelData = dicomFile.pixelData else {
            return DICOMConversionResult(
                image: nil,
                error: .noPixelData,
                diagnosticInfo: diagnosticInfo
            )
        }
        
        // Try to decompress if compressed
        let pixelData: Data
        if parseInfo.isCompressed {
            if let decompressedData = decompressor.decompress(
                pixelData: originalPixelData,
                metadata: metadata,
                transferSyntax: parseInfo.transferSyntax ?? ""
            ) {
                pixelData = decompressedData
            } else {
                return DICOMConversionResult(
                    image: nil,
                    error: .compressedNotSupported(parseInfo.transferSyntax ?? "unknown"),
                    diagnosticInfo: diagnosticInfo
                )
            }
        } else {
            pixelData = originalPixelData
        }
        
        // Check data size
        guard pixelData.count >= expectedSize else {
            return DICOMConversionResult(
                image: nil,
                error: .insufficientPixelData(expected: expectedSize, actual: pixelData.count),
                diagnosticInfo: diagnosticInfo
            )
        }
        
        // Check bits allocated
        guard metadata.bitsAllocated == 8 || metadata.bitsAllocated == 16 else {
            return DICOMConversionResult(
                image: nil,
                error: .unsupportedBitsAllocated(metadata.bitsAllocated),
                diagnosticInfo: diagnosticInfo
            )
        }
        
        let wc = windowCenter ?? metadata.windowCenter ?? calculateAutoWindowCenter(metadata: metadata)
        let ww = windowWidth ?? metadata.windowWidth ?? calculateAutoWindowWidth(metadata: metadata)
        
        switch createCGImage(pixelData: pixelData, metadata: metadata, windowCenter: wc, windowWidth: ww) {
        case .success(let cgImage):
            let uiImage = UIImage(cgImage: cgImage)
            return DICOMConversionResult(
                image: uiImage,
                error: nil,
                diagnosticInfo: diagnosticInfo
            )
        case .failure(let error):
            return DICOMConversionResult(
                image: nil,
                error: error,
                diagnosticInfo: diagnosticInfo
            )
        }
    }
    
    /// Convert a DICOM file to a platform image with optional window/level adjustment
    public func convert(
        _ dicomFile: DICOMFile,
        windowCenter: Double? = nil,
        windowWidth: Double? = nil
    ) -> PlatformImage? {
        convertWithDiagnostics(dicomFile, windowCenter: windowCenter, windowWidth: windowWidth).image
    }
    
    /// Create a CGImage from DICOM pixel data
    public func createCGImage(
        pixelData: Data,
        metadata: DICOMMetadata,
        windowCenter: Double,
        windowWidth: Double
    ) -> Result<CGImage, DICOMImageError> {
        let width = metadata.columns
        let height = metadata.rows
        let bitsAllocated = metadata.bitsAllocated
        let pixelRepresentation = metadata.pixelRepresentation
        
        let expectedPixelCount = width * height * metadata.samplesPerPixel
        let bytesPerPixel = bitsAllocated / 8
        
        guard pixelData.count >= expectedPixelCount * bytesPerPixel else {
            return .failure(.insufficientPixelData(
                expected: expectedPixelCount * bytesPerPixel,
                actual: pixelData.count
            ))
        }
        
        var outputData = Data(count: width * height)
        
        let isMonochrome1 = metadata.photometricInterpretation == "MONOCHROME1"
        let rescaleSlope = metadata.rescaleSlope
        let rescaleIntercept = metadata.rescaleIntercept
        
        let halfWindow = windowWidth / 2.0
        let minValue = windowCenter - halfWindow
        let maxValue = windowCenter + halfWindow
        
        pixelData.withUnsafeBytes { inputPtr in
            outputData.withUnsafeMutableBytes { outputPtr in
                guard 
                    let inputBase = inputPtr.baseAddress,
                    let outputBase = outputPtr.baseAddress 
                else {
                    return 
                }
                
                let output = outputBase.assumingMemoryBound(to: UInt8.self)
                
                for i in 0..<(width * height) {
                    let rawValue: Double
                    
                    if bitsAllocated == 16 {
                        let pixelOffset = i * 2
                        if pixelRepresentation == 1 {
                            let value = inputBase.advanced(by: pixelOffset)
                                .assumingMemoryBound(to: Int16.self).pointee
                            rawValue = Double(value)
                        } else {
                            let value = inputBase.advanced(by: pixelOffset)
                                .assumingMemoryBound(to: UInt16.self).pointee
                            rawValue = Double(value)
                        }
                    } else if bitsAllocated == 8 {
                        let value = inputBase.advanced(by: i).assumingMemoryBound(to: UInt8.self).pointee
                        rawValue = Double(value)
                    } else {
                        rawValue = 0
                    }
                    
                    let hounsfield = rawValue * rescaleSlope + rescaleIntercept
                    
                    var normalizedValue: Double
                    if hounsfield <= minValue {
                        normalizedValue = 0
                    } else if hounsfield >= maxValue {
                        normalizedValue = 1
                    } else {
                        normalizedValue = (hounsfield - minValue) / windowWidth
                    }
                    
                    if isMonochrome1 {
                        normalizedValue = 1 - normalizedValue
                    }
                    
                    output[i] = UInt8(normalizedValue * 255)
                }
            }
        }
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let provider = CGDataProvider(data: outputData as CFData) else {
            return .failure(.failedToCreateDataProvider)
        }
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            return .failure(.failedToCreateCGImage)
        }
        
        return .success(cgImage)
    }
    
    // MARK: - Auto Window/Level Calculation
    
    private func calculateAutoWindowCenter(metadata: DICOMMetadata) -> Double {
        // For CT scans, default to soft tissue window
        if metadata.modality == "CT" {
            return 40 // Soft tissue
        }
        
        // For other modalities, calculate based on bit depth
        let maxValue = pow(2.0, Double(metadata.bitsStored)) - 1
        return maxValue / 2
    }
    
    private func calculateAutoWindowWidth(metadata: DICOMMetadata) -> Double {
        // For CT scans, default to soft tissue window
        if metadata.modality == "CT" {
            return 400 // Soft tissue
        }
        
        // For other modalities, use full range
        return pow(2.0, Double(metadata.bitsStored))
    }
}

// MARK: - Window Presets

public enum DICOMWindowPreset: String, CaseIterable, Sendable {
    case lung = "Lung"
    case bone = "Bone"
    case softTissue = "Soft Tissue"
    case brain = "Brain"
    case liver = "Liver"
    case custom = "Custom"
    
    public var windowCenter: Double {
        switch self {
        case .lung: -600
        case .bone: 400
        case .softTissue: 40
        case .brain: 40
        case .liver: 60
        case .custom: 0
        }
    }
    
    public var windowWidth: Double {
        switch self {
        case .lung: 1500
        case .bone: 1500
        case .softTissue: 400
        case .brain: 80
        case .liver: 150
        case .custom: 1000
        }
    }
}
