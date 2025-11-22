// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let targets: [PackageDescription.Target] = [
    .target(
        name: "FoundationInternalImpl",
        dependencies: [
            .Core.NetworkInterface,
            .Core.NetworkImpl,
            .FoundationInternalInterface,
        ],
        path: "Sources/FoundationInternal/FoundationInternalImpl",
    ),
    .target(
        name: "FoundationInternalInterface",
        dependencies: [
            .Core.NetworkInterface,
        ],
        path: "Sources/FoundationInternal/FoundationInternalInterface",
    ),
    .target(
        name: "CoreNetworkImpl",
        dependencies: [
            .Core.NetworkInterface,
        ],
        path: "Sources/CoreNetwork/CoreNetworkImpl",
    ),
    .target(
        name: "CoreNetworkInterface",
        dependencies: [],
        path: "Sources/CoreNetwork/CoreNetworkInterface",
    ),
    .target(
        name: "CoreSwiftUI",
        dependencies: [
            .Core.NetworkImpl,
            .Core.NetworkInterface,
            .FoundationInternalInterface,
            .FoundationInternalImpl,
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

    targets: targets,
)

extension Target.Dependency {
    static let FoundationInternalImpl: Target.Dependency = .byName(name: "FoundationInternalImpl")
    static let FoundationInternalInterface: Target.Dependency = .byName(name: "FoundationInternalInterface")

    enum Core {
        static let NetworkImpl: Target.Dependency = .byName(name: "CoreNetworkImpl")
        static let NetworkInterface: Target.Dependency = .byName(name: "CoreNetworkInterface")
        static let SwiftUI: Target.Dependency = .byName(name: "CoreSwiftUI")
    }
}
