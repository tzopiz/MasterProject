//  Created by Dmitrii Korchagin on 26.11.2025.

import Foundation

// MARK: - Mock Data for Previews

#if DEBUG

public extension DICOMFile {
    /// Creates a mock DICOM file with generated pixel data
    static func mock(
        width: Int = 256,
        height: Int = 256,
        sliceIndex: Int = 0,
        pattern: MockPattern = .gradient
    ) -> DICOMFile {
        let metadata = DICOMMetadata(
            rows: height,
            columns: width,
            bitsAllocated: 16,
            bitsStored: 12,
            highBit: 11,
            pixelRepresentation: 0,
            samplesPerPixel: 1,
            photometricInterpretation: "MONOCHROME2",
            hasExplicitDimensions: true,
            windowCenter: 2048,
            windowWidth: 4096,
            rescaleSlope: 1.0,
            rescaleIntercept: 0.0,
            pixelSpacing: (0.5, 0.5),
            sliceThickness: 1.0,
            sliceLocation: Double(sliceIndex),
            imagePosition: (0, 0, Double(sliceIndex)),
            instanceNumber: sliceIndex + 1,
            patientName: "Mock Patient",
            patientID: "MOCK001",
            studyDate: "20251126",
            seriesDescription: "Mock CT Series",
            modality: "CT"
        )
        
        let pixelData = generatePixelData(
            width: width,
            height: height,
            sliceIndex: sliceIndex,
            pattern: pattern
        )
        
        let parseInfo = DICOMParseInfo(
            hasDICMPrefix: true,
            foundTags: ["(0028,0010)", "(0028,0011)", "(7FE0,0010)"],
            transferSyntax: "1.2.840.10008.1.2.1",
            isCompressed: false,
            warnings: []
        )
        
        return DICOMFile(
            url: URL(fileURLWithPath: "/mock/slice_\(sliceIndex).dcm"),
            metadata: metadata,
            pixelData: pixelData,
            parseInfo: parseInfo
        )
    }
    
    enum MockPattern {
        case gradient
        case checkerboard
        case circle
        case noise
        case skull // Simple skull-like shape
    }
    
    private static func generatePixelData(
        width: Int,
        height: Int,
        sliceIndex: Int,
        pattern: MockPattern
    ) -> Data {
        var data = Data(count: width * height * 2) // 16-bit
        
        data.withUnsafeMutableBytes { ptr in
            let pixels = ptr.bindMemory(to: UInt16.self)
            
            for y in 0..<height {
                for x in 0..<width {
                    let index = y * width + x
                    let value: UInt16
                    
                    switch pattern {
                    case .gradient:
                        // Diagonal gradient
                        let normalized = Double(x + y) / Double(width + height)
                        value = UInt16(normalized * 4095)
                        
                    case .checkerboard:
                        let size = 32
                        let isWhite = ((x / size) + (y / size)) % 2 == 0
                        value = isWhite ? 3500 : 500
                        
                    case .circle:
                        let cx = width / 2
                        let cy = height / 2
                        let radius = min(width, height) / 3
                        let dx = x - cx
                        let dy = y - cy
                        let dist = sqrt(Double(dx * dx + dy * dy))
                        if dist < Double(radius) {
                            value = UInt16(3000 - dist * 10)
                        } else {
                            value = 200
                        }
                        
                    case .noise:
                        value = UInt16.random(in: 500...3500)
                        
                    case .skull:
                        value = generateSkullValue(x: x, y: y, width: width, height: height, slice: sliceIndex)
                    }
                    
                    pixels[index] = value
                }
            }
        }
        
        return data
    }
    
    private static func generateSkullValue(x: Int, y: Int, width: Int, height: Int, slice: Int) -> UInt16 {
        let cx = Double(width) / 2
        let cy = Double(height) / 2
        let fx = Double(x)
        let fy = Double(y)
        
        // Outer skull (bone)
        let outerRadius = Double(min(width, height)) * 0.4
        let innerRadius = outerRadius * 0.85
        
        let dx = fx - cx
        let dy = fy - cy
        let dist = sqrt(dx * dx + dy * dy)
        
        // Bone ring
        if dist >= innerRadius && dist <= outerRadius {
            return 3500 // Bone density
        }
        
        // Brain tissue inside
        if dist < innerRadius {
            // Add some texture
            let noise = sin(fx * 0.1) * cos(fy * 0.1) * 200
            return UInt16(max(0, min(4095, 1500 + Int(noise))))
        }
        
        // Air/background outside
        return 100
    }
}

public extension DICOMSeries {
    /// Creates a mock series with multiple slices
    static func mock(
        sliceCount: Int = 12,
        width: Int = 256,
        height: Int = 256,
        pattern: DICOMFile.MockPattern = .skull
    ) -> DICOMSeries {
        let files = (0..<sliceCount).map { index in
            DICOMFile.mock(
                width: width,
                height: height,
                sliceIndex: index,
                pattern: pattern
            )
        }
        return DICOMSeries(files: files)
    }
}

#endif

