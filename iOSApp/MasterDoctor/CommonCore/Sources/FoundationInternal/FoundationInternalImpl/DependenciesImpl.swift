//  Created by Dmitrii Korchagin on 22.11.2025.

import CoreNetworkInterface
import FoundationInternalInterface

public struct DependenciesImpl: Dependencies {
    public let networkingService: any NetworkingServiceProtocol

    public init(
        networkingService: any NetworkingServiceProtocol,
    ) {
        self.networkingService = networkingService
    }
}
