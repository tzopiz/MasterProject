// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let targets: [PackageDescription.Target] = [
    .target(
        name: "FoundationInternal",
        dependencies: [

        ],
        path: "Sources/FoundationInternal",
    ),
]

let package = Package(
    name: "Foundation",
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
    enum Core {
        static let Network: Target.Dependency = .product(name: "CoreNetwork", package: "CommonCore")
        static let SwiftUI: Target.Dependency = .product(name: "CoreSwiftUI", package: "CommonCore")
        static let CommonDependencies: Target.Dependency = .product(name: "CommonDependencies", package: "CommonCore")
    }
}
