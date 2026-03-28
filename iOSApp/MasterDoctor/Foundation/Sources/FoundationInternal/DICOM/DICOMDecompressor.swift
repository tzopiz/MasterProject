//  Created by Dmitrii Korchagin on 26.11.2025.

import Foundation
import ImageIO
import CoreGraphics

/// Handles decompression of compressed DICOM pixel data
public struct DICOMDecompressor: Sendable {
    
    public init() {}
    
    /// Decompress encapsulated pixel data
    public func decompress(
        pixelData: Data,
        metadata: DICOMMetadata,
        transferSyntax: String
    ) -> Data? {
        let expectedSize = metadata.columns * metadata.rows * (metadata.bitsAllocated / 8)
        
        #if DEBUG
        print("🔍 DICOMDecompressor: Expected size = \(expectedSize), pixel data size = \(pixelData.count)")
        print("🔍 Dimensions: \(metadata.columns)x\(metadata.rows), bits: \(metadata.bitsAllocated)")
        print("🔍 First 32 bytes: \(pixelData.prefix(32).map { String(format: "%02X", $0) }.joined(separator: " "))")
        #endif
        
        // Strategy 1: Try to extract encapsulated frames
        let frames = extractFrames(from: pixelData)
        
        #if DEBUG
        print("🔍 Extracted \(frames.count) encapsulated frames")
        #endif
        
        // Check frames for exact match
        for (i, frame) in frames.enumerated() {
            if frame.count == expectedSize {
                #if DEBUG
                print("✅ Frame \(i) has exact expected size")
                #endif
                return frame
            }
        }
        
        // Strategy 2: Find JPEG data directly by marker
        if let jpegData = findJPEGData(in: pixelData) {
            #if DEBUG
            print("🔍 Found JPEG data: \(jpegData.count) bytes")
            #endif
            
            if let decoded = decodeJPEG(jpegData, metadata: metadata) {
                return decoded
            }
        }
        
        // Strategy 3: Try combined frames
        let allFrameData = frames.reduce(Data()) { $0 + $1 }
        if !allFrameData.isEmpty {
            if let jpegData = findJPEGData(in: allFrameData) {
                if let decoded = decodeJPEG(jpegData, metadata: metadata) {
                    return decoded
                }
            }
        }
        
        // Strategy 4: Try the whole data as JPEG
        if looksLikeJPEGData(pixelData) {
            #if DEBUG
            print("🔍 Pixel data starts with JPEG marker, trying direct decode")
            #endif
            if let decoded = decodeJPEG(pixelData, metadata: metadata) {
                return decoded
            }
        }
        
        #if DEBUG
        print("❌ Decompression failed - JPEG Lossless not supported by iOS")
        #endif
        return nil
    }
    
    // MARK: - JPEG Detection and Extraction
    
    /// Find JPEG data by looking for SOI marker (FF D8)
    private func findJPEGData(in data: Data) -> Data? {
        // Look for JPEG Start of Image marker
        var startIndex: Int?
        
        for i in 0..<(data.count - 1) {
            if data[i] == 0xFF && data[i + 1] == 0xD8 {
                startIndex = i
                #if DEBUG
                print("🔍 Found JPEG SOI at offset \(i)")
                #endif
                break
            }
        }
        
        guard let start = startIndex else {
            #if DEBUG
            print("🔍 No JPEG SOI marker found")
            #endif
            return nil
        }
        
        // Look for End of Image marker (FF D9) from the end
        var endIndex: Int?
        for i in stride(from: data.count - 2, through: start, by: -1) {
            if data[i] == 0xFF && data[i + 1] == 0xD9 {
                endIndex = i + 2
                break
            }
        }
        
        if let end = endIndex {
            return data.subdata(in: start..<end)
        } else {
            // No EOI found, return from start to end
            return data.subdata(in: start..<data.count)
        }
    }
    
    // MARK: - Frame Extraction (Encapsulated format)
    
    private func extractFrames(from data: Data) -> [Data] {
        var frames: [Data] = []
        var offset = 0
        var isFirstItem = true
        
        while offset + 8 <= data.count {
            let group = readUInt16(from: data, at: offset)
            let element = readUInt16(from: data, at: offset + 2)
            let length = readUInt32(from: data, at: offset + 4)
            
            // Check for item tag (FFFE, E000)
            if group == 0xFFFE && element == 0xE000 {
                offset += 8
                let itemLength = Int(length)
                
                if itemLength > 0 && itemLength != Int(0xFFFFFFFF) && offset + itemLength <= data.count {
                    if isFirstItem {
                        isFirstItem = false
                        // Skip Basic Offset Table
                    } else {
                        let itemData = data.subdata(in: offset..<offset+itemLength)
                        frames.append(itemData)
                    }
                    offset += itemLength
                } else if itemLength == 0 {
                    isFirstItem = false
                }
            } else if group == 0xFFFE && element == 0xE0DD {
                // Sequence delimitation
                break
            } else {
                // Not encapsulated format
                break
            }
        }
        
        return frames
    }
    
    // MARK: - JPEG Decoding
    
    private func decodeJPEG(_ jpegData: Data, metadata: DICOMMetadata) -> Data? {
        #if DEBUG
        print("🔍 Attempting to decode JPEG (\(jpegData.count) bytes)")
        print("🔍 JPEG header: \(jpegData.prefix(16).map { String(format: "%02X", $0) }.joined(separator: " "))")
        #endif
        
        let options: [CFString: Any] = [
            kCGImageSourceShouldCache: false
        ]
        
        guard let imageSource = CGImageSourceCreateWithData(jpegData as CFData, options as CFDictionary) else {
            #if DEBUG
            print("❌ CGImageSourceCreateWithData failed")
            #endif
            return nil
        }
        
        guard let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            #if DEBUG
            print("❌ CGImageSourceCreateImageAtIndex failed")
            #endif
            return nil
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let bitsPerComponent = cgImage.bitsPerComponent
        let bitsPerPixel = cgImage.bitsPerPixel
        let bytesPerRow = cgImage.bytesPerRow
        
        #if DEBUG
        print("✅ JPEG decoded: \(width)x\(height)")
        print("🔍 CGImage: bpc=\(bitsPerComponent), bpp=\(bitsPerPixel), bytesPerRow=\(bytesPerRow)")
        #endif
        
        // Get raw data from provider
        guard let dataProvider = cgImage.dataProvider,
              let cfData = dataProvider.data else {
            #if DEBUG
            print("❌ Failed to get data from provider")
            #endif
            return nil
        }
        
        let rawData = cfData as Data
        
        #if DEBUG
        print("🔍 Raw data size: \(rawData.count)")
        print("🔍 Expected if contiguous 16-bit: \(width * height * 2)")
        print("🔍 Expected with bytesPerRow: \(bytesPerRow * height)")
        #endif
        
        // Create output: 16-bit grayscale, width * height * 2 bytes
        let outputSize = width * height * 2
        var outputData = Data(count: outputSize)
        
        // Calculate actual source bytes per pixel from bytesPerRow
        let srcBytesPerPixel = bytesPerRow / width
        
        #if DEBUG
        print("🔍 Actual bytes per pixel from bytesPerRow: \(srcBytesPerPixel)")
        #endif
        
        outputData.withUnsafeMutableBytes { dstPtr in
            rawData.withUnsafeBytes { srcPtr in
                let dst = dstPtr.bindMemory(to: UInt16.self)
                let src = srcPtr.bindMemory(to: UInt8.self)
                
                // First pass: find min/max (reading 16-bit values with correct stride)
                var minVal: UInt16 = .max
                var maxVal: UInt16 = .min
                
                for y in 0..<height {
                    for x in 0..<width {
                        let srcOffset = y * bytesPerRow + x * srcBytesPerPixel
                        if srcOffset + 1 < rawData.count {
                            // Read 16-bit value (little endian)
                            let value = UInt16(src[srcOffset]) | (UInt16(src[srcOffset + 1]) << 8)
                            minVal = min(minVal, value)
                            maxVal = max(maxVal, value)
                        }
                    }
                }
                
                let range = Double(maxVal) - Double(minVal)
                
                #if DEBUG
                print("🔍 Pixel value range: \(minVal) - \(maxVal)")
                #endif
                
                // Second pass: normalize and copy
                for y in 0..<height {
                    for x in 0..<width {
                        let srcOffset = y * bytesPerRow + x * srcBytesPerPixel
                        let dstIdx = y * width + x
                        
                        if srcOffset + 1 < rawData.count && dstIdx < dst.count {
                            // Read 16-bit value (little endian)
                            let value = UInt16(src[srcOffset]) | (UInt16(src[srcOffset + 1]) << 8)
                            
                            // Normalize to 0-4095
                            let normalized = range > 0 ? (Double(value) - Double(minVal)) / range : 0
                            dst[dstIdx] = UInt16(normalized * 4095)
                        }
                    }
                }
            }
        }
        
        return outputData
    }
    
    private func extractPixelData(from cgImage: CGImage, targetWidth: Int, targetHeight: Int, targetBits: Int) -> Data? {
        let width = cgImage.width
        let height = cgImage.height
        let bitsPerComponent = cgImage.bitsPerComponent
        let bitsPerPixel = cgImage.bitsPerPixel
        
        #if DEBUG
        print("🔍 CGImage: \(width)x\(height), bpc=\(bitsPerComponent), bpp=\(bitsPerPixel)")
        print("🔍 ColorSpace: \(cgImage.colorSpace?.name ?? "nil" as CFString)")
        #endif
        
        // Try to get data directly from CGImage's data provider
        if let dataProvider = cgImage.dataProvider,
           let cfData = dataProvider.data {
            let rawData = cfData as Data
            
            #if DEBUG
            print("🔍 Raw data from provider: \(rawData.count) bytes")
            #endif
            
            // If it's 16-bit grayscale and matches our expected size
            let expectedSize16 = width * height * 2
            if rawData.count == expectedSize16 && targetBits == 16 {
                #if DEBUG
                print("✅ Using raw 16-bit data directly")
                #endif
                
                // Normalize 16-bit data to 12-bit range (0-4095)
                // ImageIO decodes to full 16-bit range, but DICOM expects 12-bit
                var normalizedData = Data(count: rawData.count)
                
                rawData.withUnsafeBytes { src in
                    normalizedData.withUnsafeMutableBytes { dst in
                        let srcPtr = src.bindMemory(to: UInt16.self)
                        let dstPtr = dst.bindMemory(to: UInt16.self)
                        
                        // Find actual min/max
                        var minVal: UInt16 = .max
                        var maxVal: UInt16 = .min
                        for i in 0..<srcPtr.count {
                            minVal = min(minVal, srcPtr[i])
                            maxVal = max(maxVal, srcPtr[i])
                        }
                        
                        #if DEBUG
                        print("🔍 Pixel range: min=\(minVal), max=\(maxVal)")
                        #endif
                        
                        // Normalize to 0-4095 range
                        let range = Double(maxVal) - Double(minVal)
                        if range > 0 {
                            for i in 0..<srcPtr.count {
                                let normalized = (Double(srcPtr[i]) - Double(minVal)) / range
                                dstPtr[i] = UInt16(normalized * 4095)
                            }
                        } else {
                            for i in 0..<srcPtr.count {
                                dstPtr[i] = srcPtr[i]
                            }
                        }
                    }
                }
                
                #if DEBUG
                print("✅ Normalized to 12-bit range")
                #endif
                
                return normalizedData
            }
            
            // If source is 16-bit RGB/RGBA, extract grayscale
            if bitsPerComponent == 16 {
                return extract16BitGrayscale(from: rawData, width: width, height: height, bitsPerPixel: bitsPerPixel)
            }
            
            // If source is 8-bit
            if bitsPerComponent == 8 {
                return extract8BitGrayscale(from: rawData, width: width, height: height, bitsPerPixel: bitsPerPixel, targetBits: targetBits)
            }
        }
        
        // Fallback: render to 8-bit context
        #if DEBUG
        print("🔍 Fallback: rendering to 8-bit context")
        #endif
        
        return renderTo8BitGrayscale(cgImage: cgImage, targetBits: targetBits)
    }
    
    private func extract16BitGrayscale(from data: Data, width: Int, height: Int, bitsPerPixel: Int) -> Data? {
        let componentsPerPixel = bitsPerPixel / 16
        var result = Data(count: width * height * 2)
        
        #if DEBUG
        print("🔍 Extracting 16-bit grayscale, components per pixel: \(componentsPerPixel)")
        #endif
        
        data.withUnsafeBytes { src in
            result.withUnsafeMutableBytes { dst in
                guard let srcPtr = src.baseAddress?.assumingMemoryBound(to: UInt16.self),
                      let dstPtr = dst.baseAddress?.assumingMemoryBound(to: UInt16.self) else {
                    return
                }
                
                for i in 0..<(width * height) {
                    if componentsPerPixel == 1 {
                        // Already grayscale
                        dstPtr[i] = srcPtr[i]
                    } else {
                        // RGB - take first component or average
                        let srcIdx = i * componentsPerPixel
                        dstPtr[i] = srcPtr[srcIdx] // Just take R channel
                    }
                }
            }
        }
        
        return result
    }
    
    private func extract8BitGrayscale(from data: Data, width: Int, height: Int, bitsPerPixel: Int, targetBits: Int) -> Data? {
        let componentsPerPixel = bitsPerPixel / 8
        
        #if DEBUG
        print("🔍 Extracting 8-bit grayscale, components per pixel: \(componentsPerPixel)")
        #endif
        
        var gray8bit = Data(count: width * height)
        
        data.withUnsafeBytes { src in
            gray8bit.withUnsafeMutableBytes { dst in
                guard let srcPtr = src.baseAddress?.assumingMemoryBound(to: UInt8.self),
                      let dstPtr = dst.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                    return
                }
                
                for i in 0..<(width * height) {
                    if componentsPerPixel == 1 {
                        dstPtr[i] = srcPtr[i]
                    } else {
                        let srcIdx = i * componentsPerPixel
                        dstPtr[i] = srcPtr[srcIdx]
                    }
                }
            }
        }
        
        if targetBits == 16 {
            return upscaleTo16Bit(gray8bit, width: width, height: height)
        }
        
        return gray8bit
    }
    
    private func upscaleTo16Bit(_ data8bit: Data, width: Int, height: Int) -> Data {
        var data16bit = Data(count: width * height * 2)
        
        data8bit.withUnsafeBytes { src in
            data16bit.withUnsafeMutableBytes { dst in
                guard let srcPtr = src.baseAddress?.assumingMemoryBound(to: UInt8.self),
                      let dstPtr = dst.baseAddress?.assumingMemoryBound(to: UInt16.self) else {
                    return
                }
                
                for i in 0..<(width * height) {
                    dstPtr[i] = UInt16(srcPtr[i]) << 4
                }
            }
        }
        
        return data16bit
    }
    
    private func renderTo8BitGrayscale(cgImage: CGImage, targetBits: Int) -> Data? {
        let width = cgImage.width
        let height = cgImage.height
        
        var data8bit = Data(count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        let success = data8bit.withUnsafeMutableBytes { ptr -> Bool in
            guard let baseAddress = ptr.baseAddress else { return false }
            
            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else {
                return false
            }
            
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        
        guard success else { return nil }
        
        if targetBits == 16 {
            return upscaleTo16Bit(data8bit, width: width, height: height)
        }
        
        return data8bit
    }
    
    // MARK: - Helpers
    
    private func looksLikeJPEGData(_ data: Data) -> Bool {
        guard data.count >= 2 else { return false }
        return data[0] == 0xFF && data[1] == 0xD8
    }
    
    private func readUInt16(from data: Data, at offset: Int) -> UInt16 {
        guard offset + 2 <= data.count else { return 0 }
        return data.subdata(in: offset..<offset+2).withUnsafeBytes { $0.load(as: UInt16.self) }
    }
    
    private func readUInt32(from data: Data, at offset: Int) -> UInt32 {
        guard offset + 4 <= data.count else { return 0 }
        return data.subdata(in: offset..<offset+4).withUnsafeBytes { $0.load(as: UInt32.self) }
    }
}
