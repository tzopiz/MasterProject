//  Created by Dmitrii Korchagin on 22.11.2025.

import FoundationInternalInterface
import CoreNetworkInterface
import CoreNetworkImpl

public struct FakeDependenciesImpl: Dependencies {
    public let networkingService: any NetworkingServiceProtocol

    public init() {
        self.networkingService = MockNetworkingService()
    }
}
