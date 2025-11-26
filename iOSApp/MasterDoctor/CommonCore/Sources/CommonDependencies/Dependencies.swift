//  Created by Dmitrii Korchagin on 22.11.2025.

import CoreNetwork
import FoundationInternal

public protocol Dependencies: Sendable {
    var networkingService: any NetworkingServiceProtocol { get }
    var decoder: any JSONDecoderProtocol { get }
}

public struct DependenciesImpl: Dependencies {
    public let networkingService: any NetworkingServiceProtocol
    public let decoder: any JSONDecoderProtocol

    public init(
        networkingService: any NetworkingServiceProtocol,
        decoder: any JSONDecoderProtocol,
    ) {
        self.networkingService = networkingService
        self.decoder = decoder
    }
}
