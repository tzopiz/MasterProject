//  Created by Dmitrii Korchagin on 22.11.2025.

import SwiftUI
import CoreSwiftUI
import FoundationInternalInterface
import FoundationInternalImpl
import CoreNetworkImpl

import AnalyticsApp

@main
struct MasterDoctorApp: App {
    var body: some Scene {
        WindowGroup {
            AnalysisResultView()
                .environment(\.deps, confgiureDeps())
        }
    }
}

fileprivate func confgiureDeps() -> any Dependencies {
    DependenciesImpl(
        networkingService: NetworkingService(),
    )
}
