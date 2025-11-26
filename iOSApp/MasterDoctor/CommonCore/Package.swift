// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let targets: [PackageDescription.Target] = [
    .target(
        name: "CommonDependencies",
        dependencies: [
            .Internal.CoreNetwork,
            .Foundation.FoundationInternal,
        ],
        path: "Sources/CommonDependencies",
    ),
    .target(
        name: "CoreNetwork",
        dependencies: [
            .Foundation.FoundationInternal,
        ],
        path: "Sources/CoreNetwork",
    ),
    .target(
        name: "CoreSwiftUI",
        dependencies: [
            .Internal.CoreNetwork,
            .Internal.CommonDependencies,
            .Foundation.FoundationInternal,
        ],
        path: "Sources/CoreSwiftUI",
    ),
]

let package = Package(
    name: "CommonCore",
    defaultLocalization: "ru",
    platforms: [
        .iOS(.v18),
    ],

    // MARK: - Products
    products: targets
        .map { target -> PackageDescription.Product in
                .library(name: target.name, targets: [target.name])
        },

    dependencies: [
        .package(path: "../Foundation"),
    ],

    targets: targets,
)

extension Target.Dependency {
    enum Internal {
        static let CoreNetwork: Target.Dependency = .byName(name: "CoreNetwork")
        static let CoreSwiftUI: Target.Dependency = .byName(name: "CoreSwiftUI")
        static let CommonDependencies: Target.Dependency = .byName(name: "CommonDependencies")
    }

    enum Foundation {
        static let FoundationInternal: Target.Dependency = .product(name: "FoundationInternal", package: "Foundation")
    }
}
