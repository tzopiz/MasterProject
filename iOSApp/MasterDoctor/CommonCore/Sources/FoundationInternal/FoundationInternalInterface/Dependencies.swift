//  Created by Dmitrii Korchagin on 22.11.2025.

import CoreNetworkInterface

public protocol Dependencies: Sendable {
    var networkingService: any NetworkingServiceProtocol { get }
}
