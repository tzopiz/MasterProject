//  Created by Dmitrii Korchagin on 22.11.2025.

import FoundationInternalInterface
import FoundationInternalImpl
import SwiftUI

private struct Deps: EnvironmentKey {
    static let defaultValue: Dependencies = FakeDependenciesImpl()
}

public extension EnvironmentValues {
    var deps: Dependencies {
        get { self[Deps.self] }
        set { self[Deps.self] = newValue }
    }
}
