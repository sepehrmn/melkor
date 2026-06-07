// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DA3CoreML",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "DA3CoreML",
            targets: ["DA3CoreML"]
        ),
        .executable(
            name: "da3-coreml",
            targets: ["DA3CLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "DA3CoreML",
            path: "Sources/DA3CoreML",
            swiftSettings: [
                // Enable the non-deprecated Accelerate LAPACK interface (uses `__LAPACK_int`).
                .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK"]),
            ]
        ),
        .executableTarget(
            name: "DA3CLI",
            dependencies: [
                "DA3CoreML",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/DA3CLI"
        ),
        .testTarget(
            name: "DA3CoreMLTests",
            dependencies: ["DA3CoreML"],
            path: "Tests/DA3CoreMLTests"
        ),
    ]
)
