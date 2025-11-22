// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let targets: [PackageDescription.Target] = [
    .target(
        name: "AnalyticsApp",
        dependencies: [
            .Core.NetworkInterface,
            .Core.SwiftUI,
        ],
        path: "Sources/AnalyticsApp",
    ),
]

let package = Package(
    name: "MainFeatures",
    defaultLocalization: "ru",
    platforms: [
        .iOS(.v18),
    ],

    products: targets
        .map { target -> PackageDescription.Product in
                .library(name: target.name, targets: [target.name])
        },

    dependencies: [
        .package(path: "../CommonCore"),
    ],

    targets: targets,
)

extension Target.Dependency {
    static let FoundationInternalImpl: Target.Dependency = .product(name: "FoundationInternalImpl", package: "CommonCore")
    static let FoundationInternalInterface: Target.Dependency = .product(name: "FoundationInternalInterface", package: "CommonCore")

    enum Core {
        static let NetworkImpl: Target.Dependency = .product(name: "CoreNetworkImpl", package: "CommonCore")
        static let NetworkInterface: Target.Dependency = .product(name: "CoreNetworkInterface", package: "CommonCore")
        static let SwiftUI: Target.Dependency = .product(name: "CoreSwiftUI", package: "CommonCore")
    }
}
