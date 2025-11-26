//  Created by Dmitrii Korchagin on 22.11.2025.

import SwiftUI
import CoreSwiftUI
import CommonDependencies
import FoundationInternal
import CoreNetwork

import AnalyticsApp

@main
struct MasterDoctorApp: App {
    var body: some Scene {
        WindowGroup {
            TMJDetectionView()
                .environment(\.deps, confgiureDeps())
        }
    }
}

fileprivate func confgiureDeps() -> any Dependencies {
    let decoder = JSONDecoderService()

    return DependenciesImpl(
        networkingService: NetworkingService(decoder: decoder),
        decoder: decoder,
    )
}
