//  Created by Dmitrii Korchagin on 22.11.2025.

import CoreNetwork
import FoundationInternal

public struct FakeDependenciesImpl: Dependencies {
    public let networkingService: any NetworkingServiceProtocol
    public let decoder: any JSONDecoderProtocol

    public init() {
        self.networkingService = MockNetworkingService()
        self.decoder = JSONDecoderService()
    }
}
