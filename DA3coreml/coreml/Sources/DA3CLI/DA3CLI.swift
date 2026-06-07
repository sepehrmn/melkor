import Foundation
import ArgumentParser
import DA3CoreML
import CoreML
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import simd

@available(macOS 14.0, *)
private enum CLIImageLoader {
    /// Load an image from disk with a fallback decoder for JPEGs that CoreGraphics fails to render.
    static func loadImage(from path: String) -> CGImage? {
        guard let cg = loadWithCoreGraphics(path: path) else { return nil }

        // Some real-world JPEGs (notably certain Samsung/Android camera exports) can load via ImageIO
        // but then render as an all-zero buffer when drawn. That makes the ML pipeline effectively
        // see a blank image and produce identical depth for every input.
        if canRenderNonBlank(cg) {
            return cg
        }

        // Fallback: use ffmpeg to decode to RGBA and build a bitmap-backed CGImage.
        if let decoded = decodeWithFFmpeg(path: path, width: cg.width, height: cg.height) {
            // If ffmpeg succeeded, use it regardless of content (even an actually-black image is valid).
            return decoded
        }

        return cg
    }

    private static func loadWithCoreGraphics(path: String) -> CGImage? {
        let url = URL(fileURLWithPath: path)
        let options: [CFString: Any] = [
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceShouldCache: true,
        ]
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, options as CFDictionary) else {
            return nil
        }
        return image
    }

    /// Draw to a small buffer and check whether *any* byte is non-zero.
    /// This is a cheap “is it blank?” smoke test that catches CoreGraphics decode failures.
    private static func canRenderNonBlank(_ image: CGImage) -> Bool {
        let w = 64
        let h = 64
        var buffer = [UInt8](repeating: 0, count: w * h * 4)
        let rect = CGRect(x: 0, y: 0, width: w, height: h)
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue

        buffer.withUnsafeMutableBytes { rawBuf in
            guard let base = rawBuf.baseAddress,
                  let ctx = CGContext(
                    data: base,
                    width: w,
                    height: h,
                    bitsPerComponent: 8,
                    bytesPerRow: w * 4,
                    space: cs,
                    bitmapInfo: bitmapInfo
                  ) else {
                return
            }
            ctx.interpolationQuality = .low
            ctx.draw(image, in: rect)
        }

        return buffer.contains { $0 != 0 }
    }

    private static func decodeWithFFmpeg(path: String, width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0 else { return nil }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-noautorotate",
            "-i", path,
            "-frames:v", "1",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-"
        ]

        let outPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = FileHandle.nullDevice

        do {
            try proc.run()
        } catch {
            return nil
        }

        let rgba = outPipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()

        guard proc.terminationStatus == 0 else {
            return nil
        }

        let expected = width * height * 4
        guard rgba.count == expected else {
            // If dimensions don’t match (e.g. autorotate), skip. Caller will fall back to CoreGraphics.
            return nil
        }

        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let provider = CGDataProvider(data: rgba as CFData) else { return nil }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: cs,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}

@available(macOS 14.0, *)
@main
struct DA3CLI: ParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "da3-coreml",
        abstract: "Depth-Anything-3 CoreML - Monocular depth and ray estimation",
        version: "1.0.0",
        subcommands: [Infer.self, Convert.self, Benchmark.self, To3DGS.self, Fuse.self, Stream.self],
        defaultSubcommand: Infer.self
    )
}

// MARK: - Fuse Command (multi-view to single PLY)

@available(macOS 14.0, *)
struct Fuse: ParsableCommand {
    static var configuration = CommandConfiguration(
        abstract: "Fuse multiple DA3 inputs into one 3DGS PLY using predicted poses"
    )

    @Option(name: .shortAndLong, help: "Path to backbone CoreML model (.mlmodelc)")
    var backbone: String

    @Option(name: .shortAndLong, help: "Path to DualDPT head CoreML model")
    var head: String

    @Option(name: .long, help: "Path to camera decoder CoreML model")
    var camdec: String?

    @Option(name: .long, help: "Path to GS head CoreML model (enables feed-forward Gaussian splats)")
    var gshead: String?

    @Flag(name: .long, help: "Allow depth-only fallback 3DGS fusion when --gshead is not provided (lower quality; no feed-forward GS parameters)")
    var allowDepthOnly: Bool = false

    @Flag(name: .long, help: "Use ray-based pose/intrinsics estimation (DA3 use_ray_pose) instead of camdec (GSHead mode only)")
    var useRayPose: Bool = false

    @Flag(name: .long, help: "Force the DualDPT head to run on CPU only (recommended for float32 heads)")
    var headCpuOnly: Bool = false

    @Option(name: .long, help: "Postprocess backend for DA3CoreML (cpu or metal, default: cpu). Metal accelerates crop/resize + tile blending.")
    var postprocessBackend: String = "cpu"

    @Option(name: .long, help: "Confidence activation for logits heads: linear, expp1, softplus1 (default: linear). Use when you exported `conf_activation=linear`.")
    var confidenceActivation: String = "linear"

    @Option(name: .long, help: "Confidence logit clamp min (used when confidence activation != linear, default: -30)")
    var confidenceLogitClampMin: Float = -30.0

    @Option(name: .long, help: "Confidence logit clamp max (used when confidence activation != linear, default: 30)")
    var confidenceLogitClampMax: Float = 30.0

    @Option(name: .long, help: "GS subsample factor (default: 4)")
    var gsSubsample: Int = 4

    @Option(name: .long, help: "Minimum GS confidence threshold (default: 0.0)")
    var gsMinConfidence: Float = 0.0

    @Flag(name: .long, help: "Disable GSHead offset_depth (channel 36) and use base DualDPT depth only")
    var gsDisableOffsetDepth: Bool = false

    @Option(name: .long, help: "Scale factor applied to GSHead offset_depth before adding to depth (default: 1.0)")
    var gsOffsetDepthScale: Float = 1.0

    @Flag(name: .long, help: "Use Metal for GS unprojection (depth+offsets -> world XYZ) in float32 (faster for dense splats)")
    var gsMetalUnprojection: Bool = false

    @Argument(help: "Input image path(s)")
    var inputs: [String]

    @Option(name: .shortAndLong, help: "Output fused PLY")
    var output: String = "./fused_scene.ply"

    @Option(name: .long, help: "Model size: small, base, large, giant")
    var modelSize: String = "base"

    @Option(name: .long, help: "Input image size (default: 518)")
    var inputSize: Int = 518

    @Flag(name: .long, help: "Verbose output")
    var verbose: Bool = false

    func run() throws {
        guard let size = DA3CoreML.ModelSize(rawValue: modelSize) else {
            throw ValidationError("Invalid model size: \(modelSize)")
        }

        guard let confAct = DA3CoreML.ConfidenceActivation(rawValue: confidenceActivation.lowercased()) else {
            throw ValidationError("Invalid --confidence-activation: \(confidenceActivation). Use: linear, expp1, softplus1")
        }
        let clampMin = confidenceLogitClampMin
        let clampMax = confidenceLogitClampMax

        guard let backend = DA3CoreML.PostprocessBackend(rawValue: postprocessBackend.lowercased()) else {
            throw ValidationError("Invalid --postprocess-backend: \(postprocessBackend). Use: cpu, metal")
        }

        // Derive patch size from backbone metadata when available.
        var patchSize = size.patchSize
        if let meta = DINOv3CoreML.BackboneMetadata.load(fromPath: backbone), let ps = meta.patchSize {
            patchSize = ps
        }

        // Feed-forward GS mode (preferred): uses GSHead + DualDPT depth + CamDec pose.
        if let gsheadPath = gshead {
            if verbose {
                print("📦 Loading backbone/head/camdec/gshead...")
            }

            var backboneConfig = DINOv3CoreML.Config()
            backboneConfig.inputSize = inputSize
            backboneConfig.patchSize = patchSize
            backboneConfig.useGPU = true
            backboneConfig.preferNeuralEngine = false
            let backboneModel = try DINOv3CoreML(modelPath: backbone, config: backboneConfig)

            var headConfig = DualDPTCoreML.Config()
            headConfig.dimIn = size.featureDim
            headConfig.patchSize = patchSize
            headConfig.useGPU = !headCpuOnly
            headConfig.preferNeuralEngine = false
            let headModel = try DualDPTCoreML(modelPath: head, config: headConfig)

            let camdecModel: CamDecCoreML? = {
                guard !useRayPose else { return nil }
                guard let camdecPath = camdec else { return nil }
                var camConfig = CamDecCoreML.Config()
                camConfig.dimIn = size.featureDim
                let grid = inputSize / patchSize
                camConfig.numTokens = grid * grid
                return try? CamDecCoreML(modelPath: camdecPath, config: camConfig)
            }()
            if !useRayPose, camdecModel == nil {
                throw ValidationError("Missing or invalid --camdec (required unless --use-ray-pose is set)")
            }

            var gsConfig = GSHeadCoreML.Config()
            gsConfig.dimIn = size.featureDim
            gsConfig.patchSize = patchSize
            gsConfig.useGPU = true
            let gsModel = try GSHeadCoreML(modelPath: gsheadPath, config: gsConfig)

            var convConfig = DA3GSHeadTo3DGS.Config()
            convConfig.subsample = gsSubsample
            convConfig.minConfidence = gsMinConfidence
            convConfig.applyOffsetDepth = !gsDisableOffsetDepth
            convConfig.offsetDepthScale = gsOffsetDepthScale
            convConfig.useMetalUnprojection = gsMetalUnprojection
            let converter = DA3GSHeadTo3DGS(config: convConfig)

            let fusedCloud = DA3GaussianCloud()

            for path in inputs {
                guard let img = loadImage(from: path) else {
                    print("⚠️ Failed to load \(path), skipping")
                    continue
                }
                if verbose { print("📷 \(path)") }

                // IMPORTANT (DA3 convention):
                // DA3's InputProcessor applies ImageNet normalization to the RGB tensor, and the
                // GS head consumes that same normalized tensor via `images_merger`.
                let (pixelNorm, _) = try backboneModel.preprocess(image: img, normalize: true)
                let features = try backboneModel.extractFeatures(from: pixelNorm)

                // Depth at model resolution (518x518)
                let prediction = try headModel.predict(from: features)

                // Pose/intrinsics: either camdec or ray-based (DA3 use_ray_pose).
                let extr: DA3DepthTo3DGS.CameraExtrinsics
                let intr: DA3DepthTo3DGS.CameraIntrinsics
                if useRayPose {
                    let pose: DA3RayPoseEstimator.Pose
                    do {
                        let rayConfForPose: MLMultiArray
                        if confAct == .linear {
                            rayConfForPose = prediction.rayConfidence
                        } else {
                            rayConfForPose = try DA3CoreML.activateConfidence(
                                prediction.rayConfidence,
                                activation: confAct,
                                clampMin: clampMin,
                                clampMax: clampMax
                            )
                        }

                        pose = try DA3RayPoseEstimator.estimatePose(
                            rays: prediction.rays,
                            rayConfidence: rayConfForPose,
                            imageWidth: inputSize,
                            imageHeight: inputSize
                        )
                    } catch {
                        print("   ⚠️ Ray-pose failed for \(path): \(error)")
                        continue
                    }
                    let K = pose.intrinsics
                    intr = DA3DepthTo3DGS.CameraIntrinsics(
                        fx: K.columns.0.x,
                        fy: K.columns.1.y,
                        cx: K.columns.2.x,
                        cy: K.columns.2.y,
                        width: inputSize,
                        height: inputSize
                    )
                    extr = DA3DepthTo3DGS.CameraExtrinsics(c2w: pose.c2w)

                    if verbose {
                        let t = extr.translation
                        print(String(format: "   ✓ ray-pose K: fx=%.3f fy=%.3f cx=%.3f cy=%.3f", intr.fx, intr.fy, intr.cx, intr.cy))
                        print(String(format: "   ✓ ray-pose t: [%.6f %.6f %.6f]", t.x, t.y, t.z))
                    }
                } else {
                    guard let cam = camdecModel else {
                        throw ValidationError("camdec not available (required unless --use-ray-pose is set)")
                    }
                    let poseEnc = try cam.predictPose(from: features.layer11)
                    let pose = cam.decodePose(poseEnc: poseEnc, imageWidth: inputSize, imageHeight: inputSize)
                    extr = DA3DepthTo3DGS.CameraExtrinsics(c2w: pose.c2w)
                    let K = pose.intrinsics
                    intr = DA3DepthTo3DGS.CameraIntrinsics(
                        fx: K.columns.0.x,
                        fy: K.columns.1.y,
                        cx: K.columns.2.x,
                        cy: K.columns.2.y,
                        width: inputSize,
                        height: inputSize
                    )
                    if verbose {
                        let t = extr.translation
                        print(String(format: "   ✓ camdec K: fx=%.3f fy=%.3f cx=%.3f cy=%.3f", intr.fx, intr.fy, intr.cx, intr.cy))
                        print(String(format: "   ✓ camdec t: [%.6f %.6f %.6f]", t.x, t.y, t.z))
                    }
                }

                // GS params at model resolution (518x518)
                let params = try gsModel.predict(from: features, image: pixelNorm)

                let depthRange = minMaxDepth(prediction.depth)
                let offsetDepthRange = minMaxGSChannel(params.raw, channel: GSHeadCoreML.GSParams.Channel.offsetDepth)
                let confRange = minMaxGSChannel(params.raw, channel: GSHeadCoreML.GSParams.Channel.confidence)

                if verbose {
                    if let (dmin, dmax) = depthRange {
                        print(String(format: "   ✓ depth range (model): %.6f .. %.6f", dmin, dmax))
                    }
                    if let (omin, omax) = offsetDepthRange {
                        print(String(format: "   ✓ gs offset_depth range: %.6f .. %.6f", omin, omax))
                    }
                    if let (cmin, cmax) = confRange {
                        let label = (cmin >= 0 && cmax <= 1) ? "prob" : "logit"
                        print(String(format: "   ✓ gs conf(%@) range: %.6f .. %.6f", label, cmin, cmax))
                    }
                }

                if !gsDisableOffsetDepth,
                   gsOffsetDepthScale == 1.0,
                   let (_, dmax) = depthRange,
                   let (_, omax) = offsetDepthRange,
                   dmax.isFinite, dmax > 0,
                   omax.isFinite, omax > dmax * 50 {
                    let suggested = max(1e-6, min(1.0, dmax / omax))
                    print(String(format: "   ⚠️ gs offset_depth max (%.3f) is >> depth max (%.3f). Try `--gs-offset-depth-scale %.6f` (or `0.01`/`0.001`), or `--gs-disable-offset-depth`.", omax, dmax, suggested))
                }

                let cloud = try converter.convert(gsParams: params, depth: prediction.depth, intrinsics: intr, extrinsics: extr)
                fusedCloud.add(contentsOf: cloud.allSplats)

                if verbose {
                    print("   ✓ Added \(cloud.count) gaussians (running total: \(fusedCloud.count))")
                }
            }

            if fusedCloud.isEmpty {
                throw ValidationError("No valid inputs to fuse")
            }

            let writer = DA3PLYWriter()
            try writer.write(
                fusedCloud,
                to: output,
                format: .binary,
                comments: [
                    "mode: gshead-feedforward",
                    "pose: \(useRayPose ? "ray-pose" : "camdec")",
                    "gs_offset_depth: \(!gsDisableOffsetDepth)",
                    "gs_offset_depth_scale: \(gsOffsetDepthScale)",
                    "note: sh=dc_only",
                ]
            )
            print("✅ Fused PLY saved to: \(output)")
            return
        }

        if !allowDepthOnly {
            throw ValidationError("Missing --gshead. Depth-only fallback fusion is disabled by default because quality is much lower. Provide --gshead for feed-forward splats, or re-run with --allow-depth-only.")
        } else {
            print("⚠️ WARNING: Using depth-only fallback fusion (no GSHead). Output will be lower quality; rotations/scale are not predicted per-pixel.")
        }

        // Configure models
        var config = DA3CoreML.Config()
        config.modelSize = size
        config.inputSize = inputSize
        config.memoryLimitGB = 96.0
        config.postprocessBackend = backend
        config.confidenceActivation = confAct
        config.confidenceLogitClampMin = clampMin
        config.confidenceLogitClampMax = clampMax
        if headCpuOnly { config.headUseGPU = false }

        if verbose {
            print("📦 Loading backbone/head/camdec...")
        }

        let da3 = try DA3CoreML(backbonePath: backbone, headPath: head, config: config)
        if verbose, da3.config.postprocessBackend != config.postprocessBackend {
            print("   ⚠️ Postprocess backend fell back to \(da3.config.postprocessBackend.rawValue) (Metal unavailable)")
        }

        if useRayPose {
            throw ValidationError("--use-ray-pose is currently supported only with --gshead")
        }

        var camConfig = CamDecCoreML.Config()
        camConfig.dimIn = size.featureDim
        let grid = inputSize / patchSize
        camConfig.numTokens = grid * grid
        guard let camdecPath = camdec else {
            throw ValidationError("Missing --camdec (required unless --gshead is used)")
        }
        let camdecModel = try CamDecCoreML(modelPath: camdecPath, config: camConfig)

        let converter = DA3DepthTo3DGS()

        var views: [DA3DepthTo3DGS.ViewInput] = []

        for path in inputs {
            guard let img = loadImage(from: path) else {
                print("⚠️ Failed to load \(path), skipping")
                continue
            }
            if verbose { print("📷 \(path)" ) }

            // Features for pose
            let feats = try da3.extractBackboneFeatures(image: img).0
            let poseEnc = try camdecModel.predictPose(from: feats.layer11)
            let pose = camdecModel.decodePose(poseEnc: poseEnc, imageWidth: img.width, imageHeight: img.height)
            let extr = DA3DepthTo3DGS.CameraExtrinsics(c2w: pose.c2w)
            let K = pose.intrinsics
            let intr = DA3DepthTo3DGS.CameraIntrinsics(
                fx: K.columns.0.x,
                fy: K.columns.1.y,
                cx: K.columns.2.x,
                cy: K.columns.2.y,
                width: img.width,
                height: img.height
            )

            // Depth/ray prediction
            let pred = try da3.predict(image: img, includeRays: true)
            let depthVals = readFloatArray(pred.depth)
            let confVals = readFloatArray(pred.depthConfidence)

            let view = DA3DepthTo3DGS.ViewInput(
                depth: depthVals,
                confidence: confVals,
                colors: nil,
                width: pred.originalSize.width,
                height: pred.originalSize.height,
                intrinsics: intr,
                extrinsics: extr
            )
            views.append(view)
        }

        guard !views.isEmpty else { throw ValidationError("No valid inputs to fuse") }

        let fusedCloud = converter.convertMultiViewWorldSpace(views: views)
        let writer = DA3PLYWriter()
        try writer.write(
            fusedCloud,
            to: output,
            format: .binary,
            comments: [
                "mode: depth-only-fallback",
                "pose: camdec",
                "warning: no gshead; rotations/scale are not predicted per-pixel",
            ]
        )
        print("✅ Fused PLY saved to: \(output)")
    }

    func loadImage(from path: String) -> CGImage? {
        return CLIImageLoader.loadImage(from: path)
    }

    func readFloatArray(_ array: MLMultiArray) -> [Float] {
        // IMPORTANT: CoreML MLMultiArray may have non-contiguous memory layouts (strides/padding).
        // Use a stride-aware reader to materialize a correct row-major copy.
        if let r = try? MLMultiArrayFloatReader(array) {
            return r.readAll()
        }
        let count = array.count
        return (0..<count).map { array[$0].floatValue }
    }
}

// MARK: - Stream Command (long sequence -> camera poses + point cloud)

@available(macOS 14.0, *)
struct Stream: ParsableCommand {
    static var configuration = CommandConfiguration(
        abstract: "DA3-style streaming runner for long sequences (exports camera_poses.txt + intrinsic.txt + point cloud PLY)"
    )

    @Option(name: .shortAndLong, help: "Path to backbone CoreML model (.mlmodelc)")
    var backbone: String

    @Option(name: .shortAndLong, help: "Path to DualDPT head CoreML model")
    var head: String

    @Option(name: .long, help: "Path to camera decoder CoreML model (pose/intrinsics)")
    var camdec: String?

    @Flag(name: .long, help: "Use ray-based pose/intrinsics estimation (DA3 use_ray_pose) instead of camdec")
    var useRayPose: Bool = false

    @Option(name: .long, help: "Output directory (writes camera_poses.txt, intrinsic.txt, camera_poses.ply, pcd/*.ply)")
    var outputDir: String = "./da3_stream_output"

    @Option(name: .long, help: "Chunk size (default: 120, matches upstream da3_streaming)")
    var chunkSize: Int = 120

    @Option(name: .long, help: "Overlap between chunks (default: 60, matches upstream da3_streaming)")
    var overlap: Int = 60

    @Flag(name: .long, help: "Align chunk coordinate frames using Sim3 estimated from overlap frames (approximates upstream da3_streaming chunk alignment).")
    var alignChunks: Bool = false

    @Option(name: .long, help: "Minimum allowed Sim3 scale when --align-chunks is enabled (default: 0.3333333)")
    var sim3ScaleMin: Float = 1.0 / 3.0

    @Option(name: .long, help: "Maximum allowed Sim3 scale when --align-chunks is enabled (default: 3.0)")
    var sim3ScaleMax: Float = 3.0

    @Option(name: .long, help: "Input size for fixed-shape CoreML models (default: 518)")
    var inputSize: Int = 518

    @Option(name: .long, help: "Confidence activation for logits heads: linear, expp1, softplus1 (default: linear). Use when you exported `conf_activation=linear`.")
    var confidenceActivation: String = "linear"

    @Option(name: .long, help: "Confidence logit clamp min (used when confidence activation != linear, default: -30)")
    var confidenceLogitClampMin: Float = -30.0

    @Option(name: .long, help: "Confidence logit clamp max (used when confidence activation != linear, default: 30)")
    var confidenceLogitClampMax: Float = 30.0

    @Flag(inversion: .prefixedNo, help: "Subtract 1.0 from depth_confidence before thresholding/saving (matches upstream da3_streaming: conf = exp(x)+1).")
    var subtractConfidenceOne: Bool = true

    @Option(name: .long, help: "Point cloud sample ratio (default: 0.015, matches upstream da3_streaming)")
    var pcdSampleRatio: Double = 0.015

    @Option(name: .long, help: "Point cloud confidence threshold coefficient (threshold = mean(conf) * coef; default: 0.75, matches upstream da3_streaming)")
    var pcdConfThresholdCoef: Double = 0.75

    @Option(name: .long, help: "Random seed for point cloud sampling (default: 42)")
    var seed: UInt64 = 42

    @Option(name: .long, help: "Depth minimum for point cloud filtering (default: 0)")
    var depthMin: Float = 0.0

    @Option(name: .long, help: "Depth maximum for point cloud filtering (default: 15, matches upstream da3_streaming depth_threshold)")
    var depthMax: Float = 15.0

    @Option(name: .long, help: "Depth convention for point cloud unprojection: z (K^-1*[u,v,1]*z) or ray (unit ray * range). Default: z (matches upstream da3_streaming).")
    var depthConvention: String = "z"

    @Option(name: .long, help: "Pixel center offset added to integer pixel indices when unprojecting (default: 0.5). Use 0.0 to match exports that treat pixel centers as integer coordinates.")
    var pixelCenterOffset: Float = 0.5

    @Argument(help: "Input directory containing .jpg/.jpeg/.png frames")
    var inputDir: String

    func run() throws {
        guard chunkSize > 0 else { throw ValidationError("--chunk-size must be > 0") }
        guard overlap >= 0 else { throw ValidationError("--overlap must be >= 0") }
        guard chunkSize > overlap || overlap == 0 else {
            throw ValidationError("--chunk-size must be > --overlap (or set --overlap 0)")
        }
        guard pcdSampleRatio > 0, pcdSampleRatio <= 1.0 else {
            throw ValidationError("--pcd-sample-ratio must be in (0, 1]")
        }
        guard pcdConfThresholdCoef >= 0 else {
            throw ValidationError("--pcd-conf-threshold-coef must be >= 0")
        }
        guard sim3ScaleMin > 0, sim3ScaleMax >= sim3ScaleMin else {
            throw ValidationError("--sim3-scale-min must be > 0 and <= --sim3-scale-max")
        }
        guard depthConvention.lowercased() == "z" || depthConvention.lowercased() == "ray" else {
            throw ValidationError("--depth-convention must be 'z' or 'ray'")
        }
        guard pixelCenterOffset.isFinite, pixelCenterOffset >= 0.0, pixelCenterOffset <= 1.0 else {
            throw ValidationError("--pixel-center-offset must be in [0, 1]")
        }

        guard let confAct = DA3CoreML.ConfidenceActivation(rawValue: confidenceActivation.lowercased()) else {
            throw ValidationError("Invalid --confidence-activation: \(confidenceActivation). Use: linear, expp1, softplus1")
        }
        let clampMin = confidenceLogitClampMin
        let clampMax = confidenceLogitClampMax

        // Derive patch size from backbone metadata when available.
        var patchSize = 14
        if let meta = DINOv3CoreML.BackboneMetadata.load(fromPath: backbone), let ps = meta.patchSize {
            patchSize = ps
        }

        // Load models.
        var backboneConfig = DINOv3CoreML.Config()
        backboneConfig.inputSize = inputSize
        backboneConfig.patchSize = patchSize
        backboneConfig.useGPU = true
        backboneConfig.preferNeuralEngine = false
        let backboneModel = try DINOv3CoreML(modelPath: backbone, config: backboneConfig)

        var headConfig = DualDPTCoreML.Config()
        headConfig.patchSize = patchSize
        headConfig.useGPU = true
        headConfig.preferNeuralEngine = false
        let headModel = try DualDPTCoreML(modelPath: head, config: headConfig)

        let camdecModel: CamDecCoreML? = {
            guard !useRayPose else { return nil }
            guard let camdecPath = camdec else { return nil }
            var camConfig = CamDecCoreML.Config()
            let grid = inputSize / patchSize
            camConfig.numTokens = grid * grid
            return try? CamDecCoreML(modelPath: camdecPath, config: camConfig)
        }()
        if !useRayPose, camdecModel == nil {
            throw ValidationError("Missing or invalid --camdec (required unless --use-ray-pose is set)")
        }

        // List frames.
        let fm = FileManager.default
        let inPath = (inputDir as NSString).expandingTildeInPath
        let fileNames = try fm.contentsOfDirectory(atPath: inPath)
        let frames = fileNames
            .filter { name in
                let ext = (name as NSString).pathExtension.lowercased()
                return ext == "jpg" || ext == "jpeg" || ext == "png"
            }
            .sorted()
            .map { "\(inPath)/\($0)" }

        guard !frames.isEmpty else {
            throw ValidationError("No .jpg/.jpeg/.png files found in \(inPath)")
        }

        // Output dirs.
        let outDir = (outputDir as NSString).expandingTildeInPath
        let pcdDir = "\(outDir)/pcd"
        try fm.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        try fm.createDirectory(atPath: pcdDir, withIntermediateDirectories: true)
        try cleanupExistingStreamOutputs(outDir: outDir, pcdDir: pcdDir)

        // Chunk scheduling (matches upstream: overlap at end of each chunk; next chunk keeps overlap frames).
        let chunks = makeChunks(total: frames.count, chunkSize: chunkSize, overlap: overlap)

        // Pose/intrinsics outputs (one per frame).
        var allPoses = [simd_float4x4](repeating: matrix_identity_float4x4, count: frames.count)
        var allIntrinsics = [simd_float3x3](repeating: defaultIntrinsics(width: inputSize, height: inputSize), count: frames.count)
        var poseChunk = [Int](repeating: 0, count: frames.count)

        let plyWriter = DA3PointCloudPLYWriter()
        let depthConventionLower = depthConvention.lowercased()
        let pixelOffset = pixelCenterOffset

        for (chunkIdx, chunk) in chunks.enumerated() {
            let start = chunk.start
            let end = chunk.end
            if end <= start { continue }

            let overlapCount = max(0, min(overlap, end - start))

            var chunkInvBase = matrix_identity_float4x4
            var hasChunkInvBase = false
            var sim3 = DA3Sim3.identity

            if alignChunks, chunkIdx > 0, overlapCount > 0 {
                var sim3Cfg = DA3Sim3.EstimateConfig()
                sim3Cfg.minPointCount = 3
                sim3Cfg.estimateScale = true
                sim3Cfg.scaleClamp = sim3ScaleMin...sim3ScaleMax

                var relPoses: [simd_float4x4] = []
                var dstPoses: [simd_float4x4] = []
                relPoses.reserveCapacity(overlapCount)
                dstPoses.reserveCapacity(overlapCount)

                var srcPts: [simd_float3] = []
                var dstPts: [simd_float3] = []

                let alignEnd = min(end, start + overlapCount)
                let alignStride = max(1, overlapCount / 10)
                let alignSampleRatio = min(0.01, max(0.002, pcdSampleRatio))
                let alignSampleStep = max(1, Int(round(1.0 / sqrt(alignSampleRatio))))
                let alignStepArea = Double(alignSampleStep * alignSampleStep)
                let alignKeepProb = min(1.0, alignSampleRatio * alignStepArea)
                let maxPairsPerAlignFrame = 2048

                for idx in stride(from: start, to: alignEnd, by: alignStride) {
                    autoreleasepool {
                        let path = frames[idx]
                        do {
                            guard let img = CLIImageLoader.loadImage(from: path) else { return }

                            let (features, _, _) = try backboneModel.extractFeaturesAndPixels(from: img, normalize: true)
                            let pred = try headModel.predict(from: features)

                            let depthConf: MLMultiArray = try DA3CoreML.activateConfidence(
                                pred.depthConfidence,
                                activation: confAct,
                                clampMin: clampMin,
                                clampMax: clampMax
                            )

                            let pose: (c2w: simd_float4x4, intrinsics: simd_float3x3)
                            if useRayPose {
                                let rayConf = try DA3CoreML.activateConfidence(
                                    pred.rayConfidence,
                                    activation: confAct,
                                    clampMin: clampMin,
                                    clampMax: clampMax
                                )
                                do {
                                    let est = try DA3RayPoseEstimator.estimatePose(
                                        rays: pred.rays,
                                        rayConfidence: rayConf,
                                        imageWidth: inputSize,
                                        imageHeight: inputSize
                                    )
                                    pose = (est.c2w, est.intrinsics)
                                } catch {
                                    return
                                }
                            } else {
                                let poseEnc = try camdecModel!.predictPose(from: features.layer11)
                                pose = camdecModel!.decodePose(poseEnc: poseEnc, imageWidth: inputSize, imageHeight: inputSize)
                            }

                            let validated = validatePose(pose, inputSize: inputSize)
                            if !hasChunkInvBase {
                                chunkInvBase = validated.c2w.inverse
                                hasChunkInvBase = true
                            }

                            let chunkRel = chunkInvBase * validated.c2w
                            let dstPose = allPoses[idx]
                            relPoses.append(chunkRel)
                            dstPoses.append(dstPose)

                            // DA3-streaming-style alignment: dense (but sampled) point correspondences in the overlap.
                            let (depthFlat, h, w) = try readCHWAsHW(pred.depth)
                            let (confFlat0, _, _) = try readCHWAsHW(depthConf)

                            // Approximate the confidence median using a fixed stride sample.
                            let medianStride = max(1, (h * w) / 2048)
                            var confSamples: [Float] = []
                            confSamples.reserveCapacity((h * w) / medianStride + 1)
                            for li in stride(from: 0, to: confFlat0.count, by: medianStride) {
                                var c = confFlat0[li]
                                if !c.isFinite { continue }
                                if subtractConfidenceOne { c = max(0, c - 1.0) }
                                confSamples.append(c)
                            }
                            confSamples.sort()
                            let confMedian = confSamples.isEmpty ? 0 : confSamples[confSamples.count / 2]
                            let confThreshold = max(0, 0.1 * confMedian)

                            let srcK = validated.intrinsics
                            let dstK = allIntrinsics[idx]

                            let srcFx = srcK.columns.0.x
                            let srcFy = srcK.columns.1.y
                            let srcCx = srcK.columns.2.x
                            let srcCy = srcK.columns.2.y

                            let dstFx = dstK.columns.0.x
                            let dstFy = dstK.columns.1.y
                            let dstCx = dstK.columns.2.x
                            let dstCy = dstK.columns.2.y

                            let invSrcFx = 1.0 / max(1e-6, srcFx)
                            let invSrcFy = 1.0 / max(1e-6, srcFy)
                            let invDstFx = 1.0 / max(1e-6, dstFx)
                            let invDstFy = 1.0 / max(1e-6, dstFy)

                            var rng = SplitMix64(state: seed &+ UInt64(idx))
                            var added = 0

                            for y in stride(from: 0, to: h, by: alignSampleStep) {
                                for x in stride(from: 0, to: w, by: alignSampleStep) {
                                    if alignKeepProb < 1.0, rng.nextDouble() >= alignKeepProb { continue }
                                    let li = y * w + x
                                    let d = depthFlat[li]
                                    if !d.isFinite { continue }
                                    if d < depthMin || d > depthMax { continue }

                                    var c = confFlat0[li]
                                    if !c.isFinite { continue }
                                    if subtractConfidenceOne { c = max(0, c - 1.0) }
                                    if c <= 1e-5 || c < confThreshold { continue }

                                    let u = Float(x) + pixelOffset
                                    let v = Float(y) + pixelOffset

                                    let srcCamPoint: simd_float3
                                    let dstCamPoint: simd_float3

                                    if depthConventionLower == "z" {
                                        let srcX = (u - srcCx) * invSrcFx * d
                                        let srcY = (v - srcCy) * invSrcFy * d
                                        srcCamPoint = simd_float3(srcX, srcY, d)

                                        let dstX = (u - dstCx) * invDstFx * d
                                        let dstY = (v - dstCy) * invDstFy * d
                                        dstCamPoint = simd_float3(dstX, dstY, d)
                                    } else {
                                        let srcU = (u - srcCx) * invSrcFx
                                        let srcV = (v - srcCy) * invSrcFy
                                        var srcDir = simd_float3(srcU, srcV, 1.0)
                                        let srcLen = simd_length(srcDir)
                                        if !srcLen.isFinite || srcLen <= 0 { continue }
                                        srcDir /= srcLen
                                        srcCamPoint = srcDir * d

                                        let dstU = (u - dstCx) * invDstFx
                                        let dstV = (v - dstCy) * invDstFy
                                        var dstDir = simd_float3(dstU, dstV, 1.0)
                                        let dstLen = simd_length(dstDir)
                                        if !dstLen.isFinite || dstLen <= 0 { continue }
                                        dstDir /= dstLen
                                        dstCamPoint = dstDir * d
                                    }

                                    let srcWorld4 = chunkRel * simd_float4(srcCamPoint, 1.0)
                                    let dstWorld4 = dstPose * simd_float4(dstCamPoint, 1.0)
                                    if !srcWorld4.x.isFinite || !srcWorld4.y.isFinite || !srcWorld4.z.isFinite { continue }
                                    if !dstWorld4.x.isFinite || !dstWorld4.y.isFinite || !dstWorld4.z.isFinite { continue }

                                    srcPts.append(simd_float3(srcWorld4.x, srcWorld4.y, srcWorld4.z))
                                    dstPts.append(simd_float3(dstWorld4.x, dstWorld4.y, dstWorld4.z))
                                    added += 1
                                    if added >= maxPairsPerAlignFrame { break }
                                }
                                if added >= maxPairsPerAlignFrame { break }
                            }
                        } catch {
                            // Ignore alignment-frame failures; alignment will fall back to identity if too many fail.
                            return
                        }
                    }
                }

                // Fall back to pose-only correspondences if too few depth points survived filtering.
                if srcPts.count < sim3Cfg.minPointCount {
                    // Build point correspondences: camera centers + axis endpoints, to align both translation and rotation.
                    var axisScale: Float = 0.1
                    if dstPoses.count >= 2 {
                        var baselines: [Float] = []
                        baselines.reserveCapacity(dstPoses.count - 1)
                        for i in 1..<dstPoses.count {
                            let a = simd_float3(dstPoses[i - 1].columns.3.x, dstPoses[i - 1].columns.3.y, dstPoses[i - 1].columns.3.z)
                            let b = simd_float3(dstPoses[i].columns.3.x, dstPoses[i].columns.3.y, dstPoses[i].columns.3.z)
                            let d = simd_length(b - a)
                            if d.isFinite, d > 1e-6 { baselines.append(d) }
                        }
                        if !baselines.isEmpty {
                            baselines.sort()
                            let mid = baselines[baselines.count / 2]
                            axisScale = max(1e-3, 0.25 * mid)
                        }
                    }

                    srcPts.removeAll(keepingCapacity: true)
                    dstPts.removeAll(keepingCapacity: true)
                    srcPts.reserveCapacity(relPoses.count * 4)
                    dstPts.reserveCapacity(dstPoses.count * 4)
                    for i in 0..<min(relPoses.count, dstPoses.count) {
                        let srcPose = relPoses[i]
                        let dstPose = dstPoses[i]

                        let srcC = simd_float3(srcPose.columns.3.x, srcPose.columns.3.y, srcPose.columns.3.z)
                        let dstC = simd_float3(dstPose.columns.3.x, dstPose.columns.3.y, dstPose.columns.3.z)

                        srcPts.append(srcC)
                        dstPts.append(dstC)

                        let srcAxes = [
                            simd_float3(srcPose.columns.0.x, srcPose.columns.0.y, srcPose.columns.0.z),
                            simd_float3(srcPose.columns.1.x, srcPose.columns.1.y, srcPose.columns.1.z),
                            simd_float3(srcPose.columns.2.x, srcPose.columns.2.y, srcPose.columns.2.z),
                        ]
                        let dstAxes = [
                            simd_float3(dstPose.columns.0.x, dstPose.columns.0.y, dstPose.columns.0.z),
                            simd_float3(dstPose.columns.1.x, dstPose.columns.1.y, dstPose.columns.1.z),
                            simd_float3(dstPose.columns.2.x, dstPose.columns.2.y, dstPose.columns.2.z),
                        ]
                        for k in 0..<3 {
                            srcPts.append(srcC + srcAxes[k] * axisScale)
                            dstPts.append(dstC + dstAxes[k] * axisScale)
                        }
                    }
                }

                if let est0 = DA3Sim3.estimate(from: srcPts, to: dstPts, config: sim3Cfg) {
                    // Robustify once with a median-based inlier filter.
                    var residuals: [Float] = []
                    residuals.reserveCapacity(min(srcPts.count, dstPts.count))
                    for i in 0..<min(srcPts.count, dstPts.count) {
                        let p = est0.transformPoint(srcPts[i])
                        let q = dstPts[i]
                        let r = simd_length(p - q)
                        if r.isFinite { residuals.append(r) }
                    }
                    residuals.sort()
                    let med = residuals.isEmpty ? 0 : residuals[residuals.count / 2]
                    let thresh = max(1e-4, 5.0 * med)

                    if med.isFinite, med > 0 {
                        var inSrc: [simd_float3] = []
                        var inDst: [simd_float3] = []
                        inSrc.reserveCapacity(srcPts.count)
                        inDst.reserveCapacity(dstPts.count)
                        for i in 0..<min(srcPts.count, dstPts.count) {
                            let p = est0.transformPoint(srcPts[i])
                            let q = dstPts[i]
                            let r = simd_length(p - q)
                            if r.isFinite, r <= thresh {
                                inSrc.append(srcPts[i])
                                inDst.append(dstPts[i])
                            }
                        }
                        if let est1 = DA3Sim3.estimate(from: inSrc, to: inDst, config: sim3Cfg) {
                            sim3 = est1
                            print(String(format: "🔧 chunk %d: Sim3 aligned (scale=%.6f, pairs=%d)", chunkIdx, est1.scale, inSrc.count))
                        } else {
                            sim3 = est0
                            print(String(format: "🔧 chunk %d: Sim3 aligned (scale=%.6f, pairs=%d)", chunkIdx, est0.scale, srcPts.count))
                        }
                    } else {
                        sim3 = est0
                        print(String(format: "🔧 chunk %d: Sim3 aligned (scale=%.6f, pairs=%d)", chunkIdx, est0.scale, srcPts.count))
                    }
                } else {
                    sim3 = .identity
                    print("⚠️ chunk \(chunkIdx): Sim3 alignment failed; using identity transform")
                }
            }

            let processStart: Int
            let processEnd: Int
            if alignChunks {
                processStart = (chunkIdx == 0) ? start : min(end, start + overlapCount)
                processEnd = end
            } else {
                let saveEnd = (chunkIdx < chunks.count - 1) ? max(start, end - overlap) : end
                processStart = start
                processEnd = saveEnd
            }

            if processEnd <= processStart { continue }

            var vertexData = Data()
            var vertexCount = 0
            let bytesPerVertex = 3 * MemoryLayout<Float>.size + 3
            let expectedPointsPerFrame = Int(Double(inputSize * inputSize) * pcdSampleRatio)
            let expectedPoints = max(0, expectedPointsPerFrame) * max(0, processEnd - processStart)
            vertexData.reserveCapacity(min(256 * 1024 * 1024, expectedPoints * bytesPerVertex))

            for idx in processStart..<processEnd {
                autoreleasepool {
                    let path = frames[idx]
                    do {
                        guard let img = CLIImageLoader.loadImage(from: path) else {
                            print("⚠️ Failed to load \(path); writing identity pose and skipping point cloud for this frame")
                            allPoses[idx] = matrix_identity_float4x4
                            allIntrinsics[idx] = defaultIntrinsics(width: inputSize, height: inputSize)
                            poseChunk[idx] = chunkIdx
                            return
                        }

                        // Run backbone + head once (model-space resolution, fixed-size).
                        let (features, pixelValues, _) = try backboneModel.extractFeaturesAndPixels(from: img, normalize: true)
                        let pred = try headModel.predict(from: features)
                        let pixelReader = (try? MLMultiArrayFloatReader(pixelValues))

                        // Confidence activation (optional, for logits exports).
                        let depthConf: MLMultiArray = try DA3CoreML.activateConfidence(
                            pred.depthConfidence,
                            activation: confAct,
                            clampMin: clampMin,
                            clampMax: clampMax
                        )

                        // Pose/intrinsics.
                        let pose: (c2w: simd_float4x4, intrinsics: simd_float3x3)
                        if useRayPose {
                            let rayConf = try DA3CoreML.activateConfidence(
                                pred.rayConfidence,
                                activation: confAct,
                                clampMin: clampMin,
                                clampMax: clampMax
                            )
                            do {
                                let est = try DA3RayPoseEstimator.estimatePose(
                                    rays: pred.rays,
                                    rayConfidence: rayConf,
                                    imageWidth: inputSize,
                                    imageHeight: inputSize
                                )
                                pose = (est.c2w, est.intrinsics)
                            } catch {
                                print("⚠️ Ray-pose failed for \(path): \(error) — writing identity pose for this frame")
                                pose = (matrix_identity_float4x4, defaultIntrinsics(width: inputSize, height: inputSize))
                            }
                        } else {
                            let poseEnc = try camdecModel!.predictPose(from: features.layer11)
                            pose = camdecModel!.decodePose(poseEnc: poseEnc, imageWidth: inputSize, imageHeight: inputSize)
                        }

                        let validatedPose = validatePose(pose, inputSize: inputSize)

                        let c2wForFrame: simd_float4x4 = {
                            if alignChunks {
                                if !hasChunkInvBase {
                                    chunkInvBase = validatedPose.c2w.inverse
                                    hasChunkInvBase = true
                                }
                                let chunkRel = chunkInvBase * validatedPose.c2w
                                return sim3.transformPoseC2W(chunkRel)
                            }
                            return validatedPose.c2w
                        }()

                        allPoses[idx] = c2wForFrame
                        allIntrinsics[idx] = validatedPose.intrinsics
                        poseChunk[idx] = chunkIdx

                        // Point cloud sampling for this frame.
                        let (depthFlat, h, w) = try readCHWAsHW(pred.depth)
                        let (confFlat, _, _) = try readCHWAsHW(depthConf)
                        let rgba: [UInt8]? = pixelReader == nil ? (try? renderRGBA8(img, width: w, height: h)) : nil

                        @inline(__always)
                        func clampToByte01(_ x: Float) -> UInt8 {
                            if !x.isFinite { return 0 }
                            let clamped = min(1.0, max(0.0, x))
                            return UInt8(min(255, max(0, Int((clamped * 255.0).rounded()))))
                        }

                        @inline(__always)
                        func rgbAt(_ y: Int, _ x: Int) -> (UInt8, UInt8, UInt8)? {
                            if let r = pixelReader {
                                // pixelValues are ImageNet-normalized in CHW order (R,G,B).
                                let meanR: Float = 0.485, meanG: Float = 0.456, meanB: Float = 0.406
                                let stdR: Float = 0.229, stdG: Float = 0.224, stdB: Float = 0.225
                                let r01 = r.read(0, 0, y, x) * stdR + meanR
                                let g01 = r.read(0, 1, y, x) * stdG + meanG
                                let b01 = r.read(0, 2, y, x) * stdB + meanB
                                return (clampToByte01(r01), clampToByte01(g01), clampToByte01(b01))
                            }
                            if let rgba {
                                let li = y * w + x
                                let off = li * 4
                                guard off + 2 < rgba.count else { return nil }
                                return (rgba[off], rgba[off + 1], rgba[off + 2])
                            }
                            return nil
                        }

                        let meanConf = meanConfidence(confFlat, subtractOne: subtractConfidenceOne)
                        let confThreshold = Float(Double(meanConf) * pcdConfThresholdCoef)

                        // Fast path for small sample ratios: stride-sample the pixel grid, then optionally
                        // thin within that grid so the expected overall ratio matches `pcdSampleRatio`.
                        let sampleStep = max(1, Int(round(1.0 / sqrt(pcdSampleRatio))))
                        let stepArea = Double(sampleStep * sampleStep)
                        let keepProb = min(1.0, pcdSampleRatio * stepArea)
                        var rng = SplitMix64(state: seed &+ UInt64(idx))

                        let K = validatedPose.intrinsics
                        let fx = K.columns.0.x
                        let fy = K.columns.1.y
                        let cx = K.columns.2.x
                        let cy = K.columns.2.y

                        let invFx = 1.0 / max(1e-6, fx)
                        let invFy = 1.0 / max(1e-6, fy)

                        for y in stride(from: 0, to: h, by: sampleStep) {
                            for x in stride(from: 0, to: w, by: sampleStep) {
                                if keepProb < 1.0, rng.nextDouble() >= keepProb { continue }
                                let li = y * w + x
                                let d = depthFlat[li]
                                if !d.isFinite { continue }
                                if d < depthMin || d > depthMax { continue }

                                var c = confFlat[li]
                                if !c.isFinite { continue }
                                if subtractConfidenceOne { c = max(0, c - 1.0) }
                                if c <= 1e-5 { continue }
                                if c < confThreshold { continue }

                                let camPoint: simd_float3
                                if depthConventionLower == "z" {
                                    let u = Float(x) + pixelOffset
                                    let v = Float(y) + pixelOffset
                                    let X = (u - cx) * invFx * d
                                    let Y = (v - cy) * invFy * d
                                    camPoint = simd_float3(X, Y, d)
                                } else {
                                    let u = (Float(x) + pixelOffset - cx) * invFx
                                    let v = (Float(y) + pixelOffset - cy) * invFy
                                    var dir = simd_float3(u, v, 1.0)
                                    let len = simd_length(dir)
                                    if !len.isFinite || len <= 0 { continue }
                                    dir /= len
                                    camPoint = dir * d
                                }

                                let world4 = c2wForFrame * simd_float4(camPoint, 1.0)
                                if !world4.x.isFinite || !world4.y.isFinite || !world4.z.isFinite { continue }

                                guard let (r, g, b) = rgbAt(y, x) else { continue }

                                vertexData.appendFloat32LE(world4.x)
                                vertexData.appendFloat32LE(world4.y)
                                vertexData.appendFloat32LE(world4.z)
                                vertexData.appendUInt8(r)
                                vertexData.appendUInt8(g)
                                vertexData.appendUInt8(b)
                                vertexCount += 1
                            }
                        }
                    } catch {
                        print("⚠️ Streaming inference failed for \(path): \(error) — writing identity pose and skipping this frame")
                        allPoses[idx] = matrix_identity_float4x4
                        allIntrinsics[idx] = defaultIntrinsics(width: inputSize, height: inputSize)
                        poseChunk[idx] = chunkIdx
                    }
                }
            }

            let chunkPath = "\(pcdDir)/\(chunkIdx)_pcd.ply"
            try plyWriter.writeBinaryPointCloud(vertexCount: vertexCount, vertexData: vertexData, to: chunkPath)
            print("✅ Saved point cloud chunk: \(chunkPath) (\(vertexCount) points)")
        }

        // Export camera poses + intrinsics (DA3-Streaming compatible filenames).
        try writeCameraPosesTxt(allPoses, to: "\(outDir)/camera_poses.txt")
        try writeIntrinsicsTxt(allIntrinsics, to: "\(outDir)/intrinsic.txt")
        try writeCameraPosesPly(
            allPoses: allPoses,
            poseChunk: poseChunk,
            to: "\(outDir)/camera_poses.ply"
        )

        // Merge point clouds into pcd/combined_pcd.ply (DA3-Streaming compatible).
        try plyWriter.mergeBinaryPointCloudPLYFiles(inputDir: pcdDir, outputPath: "\(pcdDir)/combined_pcd.ply")
        print("✅ Saved merged point cloud: \(pcdDir)/combined_pcd.ply")
    }

    private func makeChunks(total: Int, chunkSize: Int, overlap: Int) -> [(start: Int, end: Int)] {
        if total <= 0 { return [] }
        if total <= chunkSize { return [(0, total)] }
        let stride = max(1, chunkSize - overlap)
        var chunks: [(Int, Int)] = []
        var start = 0
        while start < total {
            let end = min(start + chunkSize, total)
            chunks.append((start, end))
            if end == total { break }
            start += stride
        }
        return chunks
    }

    private func validatePose(
        _ pose: (c2w: simd_float4x4, intrinsics: simd_float3x3),
        inputSize: Int
    ) -> (c2w: simd_float4x4, intrinsics: simd_float3x3) {
        func mat4Finite(_ m: simd_float4x4) -> Bool {
            let cols = [m.columns.0, m.columns.1, m.columns.2, m.columns.3]
            for c in cols {
                if !c.x.isFinite || !c.y.isFinite || !c.z.isFinite || !c.w.isFinite { return false }
            }
            return true
        }

        func mat3Finite(_ m: simd_float3x3) -> Bool {
            let cols = [m.columns.0, m.columns.1, m.columns.2]
            for c in cols {
                if !c.x.isFinite || !c.y.isFinite || !c.z.isFinite { return false }
            }
            return true
        }

        let K = pose.intrinsics
        let fx = K.columns.0.x
        let fy = K.columns.1.y
        let cx = K.columns.2.x
        let cy = K.columns.2.y

        let ok = mat4Finite(pose.c2w)
            && mat3Finite(K)
            && fx.isFinite && fy.isFinite && cx.isFinite && cy.isFinite
            && fx > 1e-3 && fy > 1e-3

        if ok {
            return pose
        }

        return (matrix_identity_float4x4, defaultIntrinsics(width: inputSize, height: inputSize))
    }

    private func defaultIntrinsics(width: Int, height: Int) -> simd_float3x3 {
        let H = Float(height)
        let W = Float(width)
        let fov: Float = 50.0 * .pi / 180.0
        let fy = (H / 2.0) / max(1e-6, tan(fov / 2.0))
        let fx = (W / 2.0) / max(1e-6, tan(fov / 2.0))
        let cx = W / 2.0
        let cy = H / 2.0
        return simd_float3x3(rows: [
            simd_float3(fx, 0, cx),
            simd_float3(0, fy, cy),
            simd_float3(0, 0, 1),
        ])
    }

    private func cleanupExistingStreamOutputs(outDir: String, pcdDir: String) throws {
        let fm = FileManager.default

        func removeIfExists(_ path: String) throws {
            if fm.fileExists(atPath: path) {
                try fm.removeItem(atPath: path)
            }
        }

        try removeIfExists("\(outDir)/camera_poses.txt")
        try removeIfExists("\(outDir)/intrinsic.txt")
        try removeIfExists("\(outDir)/camera_poses.ply")

        // Remove stale PCD chunks from previous runs so merges are deterministic.
        let names = (try? fm.contentsOfDirectory(atPath: pcdDir)) ?? []
        for name in names {
            if name.hasSuffix("_pcd.ply") || name == "combined_pcd.ply" {
                try? fm.removeItem(atPath: "\(pcdDir)/\(name)")
            }
        }
    }

    private func readCHWAsHW(_ arr: MLMultiArray) throws -> (flatHW: [Float], h: Int, w: Int) {
        let r = try MLMultiArrayFloatReader(arr)
        let shape = r.shape
        let (h, w): (Int, Int) = {
            switch shape.count {
            case 4: return (shape[2], shape[3]) // [B, C, H, W]
            case 3:
                // [C, H, W] or [B, H, W]
                if shape[0] == 1 { return (shape[1], shape[2]) }
                return (shape[1], shape[2])
            case 2: return (shape[0], shape[1]) // [H, W]
            default: return (0, 0)
            }
        }()
        guard h > 0, w > 0 else { throw DA3Error.invalidInput("Unexpected tensor shape: \(shape)") }

        // Assumption: batch/channel dims are 1 when present.
        let all = r.readAll()
        if all.count == h * w { return (all, h, w) }
        // Fallback: take the first HW slice.
        return (Array(all.prefix(h * w)), h, w)
    }

    private func meanConfidence(_ conf: [Float], subtractOne: Bool) -> Float {
        var sum: Double = 0
        var n: Int = 0
        for v0 in conf {
            if !v0.isFinite { continue }
            var v = v0
            if subtractOne { v = max(0, v - 1.0) }
            sum += Double(v)
            n += 1
        }
        if n == 0 { return 0 }
        return Float(sum / Double(n))
    }

    private func writeCameraPosesTxt(_ poses: [simd_float4x4], to path: String) throws {
        let url = URL(fileURLWithPath: path)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let fh = try FileHandle(forWritingTo: url)
        defer { try? fh.close() }

        for pose in poses {
            let rowMajor = poseRowMajor(pose)
            let line = rowMajor.map { String($0) }.joined(separator: " ") + "\n"
            try fh.write(contentsOf: Data(line.utf8))
        }
    }

    private func writeIntrinsicsTxt(_ intrinsics: [simd_float3x3], to path: String) throws {
        let url = URL(fileURLWithPath: path)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let fh = try FileHandle(forWritingTo: url)
        defer { try? fh.close() }

        for K in intrinsics {
            let fx = K.columns.0.x
            let fy = K.columns.1.y
            let cx = K.columns.2.x
            let cy = K.columns.2.y
            let line = "\(fx) \(fy) \(cx) \(cy)\n"
            try fh.write(contentsOf: Data(line.utf8))
        }
    }

    private func writeCameraPosesPly(allPoses: [simd_float4x4], poseChunk: [Int], to path: String) throws {
        let url = URL(fileURLWithPath: path)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let fh = try FileHandle(forWritingTo: url)
        defer { try? fh.close() }

        let n = allPoses.count
        let header =
            "ply\n" +
            "format ascii 1.0\n" +
            "element vertex \(n)\n" +
            "property float x\n" +
            "property float y\n" +
            "property float z\n" +
            "property uchar red\n" +
            "property uchar green\n" +
            "property uchar blue\n" +
            "end_header\n"
        try fh.write(contentsOf: Data(header.utf8))

        func chunkColor(_ i: Int) -> (UInt8, UInt8, UInt8) {
            let palette: [(UInt8, UInt8, UInt8)] = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            ]
            return palette[max(0, i) % palette.count]
        }

        for (i, pose) in allPoses.enumerated() {
            let p = pose.columns.3
            let (r, g, b) = chunkColor(poseChunk[i])
            let line = "\(p.x) \(p.y) \(p.z) \(r) \(g) \(b)\n"
            try fh.write(contentsOf: Data(line.utf8))
        }
    }

    private func poseRowMajor(_ m: simd_float4x4) -> [Float] {
        let c0 = m.columns.0
        let c1 = m.columns.1
        let c2 = m.columns.2
        let c3 = m.columns.3
        return [
            c0.x, c1.x, c2.x, c3.x,
            c0.y, c1.y, c2.y, c3.y,
            c0.z, c1.z, c2.z, c3.z,
            c0.w, c1.w, c2.w, c3.w,
        ]
    }

    private func renderRGBA8(_ image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue

        try buffer.withUnsafeMutableBytes { rawBuf in
            guard let base = rawBuf.baseAddress else {
                throw DA3Error.imageProcessingFailed("Failed to allocate RGBA buffer")
            }
            guard let ctx = CGContext(
                data: base,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: cs,
                bitmapInfo: bitmapInfo
            ) else {
                throw DA3Error.imageProcessingFailed("Failed to create bitmap context")
            }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: rect)
        }

        return buffer
    }

    private struct SplitMix64 {
        var state: UInt64

        mutating func nextUInt64() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }

        mutating func nextDouble() -> Double {
            // 53-bit mantissa -> [0, 1)
            let x = nextUInt64() >> 11
            return Double(x) / Double(1 << 53)
        }
    }
}

@available(macOS 14.0, iOS 17.0, *)
private func minMaxDepth(_ depth: MLMultiArray) -> (Float, Float)? {
    guard let r = try? MLMultiArrayFloatReader(depth) else { return nil }
    let shape = r.shape
    let (h, w): (Int, Int) = {
        switch shape.count {
        case 4: return (shape[2], shape[3]) // [B, C, H, W]
        case 3: return (shape[1], shape[2]) // [C, H, W] or [B, H, W]
        case 2: return (shape[0], shape[1]) // [H, W]
        default: return (0, 0)
        }
    }()
    guard h > 0, w > 0 else { return nil }

    var minV: Float = .greatestFiniteMagnitude
    var maxV: Float = -.greatestFiniteMagnitude

    func v(_ y: Int, _ x: Int) -> Float {
        switch shape.count {
        case 4: return r.read(0, 0, y, x)
        case 3: return r.read(0, y, x)
        case 2: return r.read(y, x)
        default: return 0
        }
    }

    for y in 0..<h {
        for x in 0..<w {
            let val = v(y, x)
            guard val.isFinite else { continue }
            if val < minV { minV = val }
            if val > maxV { maxV = val }
        }
    }

    return (minV, maxV)
}

@available(macOS 14.0, *)
private extension Data {
    mutating func appendFloat32LE(_ value: Float) {
        var bits = value.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &bits) { append(contentsOf: $0) }
    }

    mutating func appendUInt8(_ value: UInt8) {
        var v = value
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }
}

@available(macOS 14.0, iOS 17.0, *)
private func minMaxGSChannel(_ gsParams: MLMultiArray, channel: Int) -> (Float, Float)? {
    guard let r = try? MLMultiArrayFloatReader(gsParams) else { return nil }
    let shape = r.shape
    // Expect [B, C, H, W]
    guard shape.count == 4, shape[1] > channel else { return nil }
    let h = shape[2], w = shape[3]
    guard h > 0, w > 0 else { return nil }

    var minV: Float = .greatestFiniteMagnitude
    var maxV: Float = -.greatestFiniteMagnitude
    for y in 0..<h {
        for x in 0..<w {
            let val = r.read(0, channel, y, x)
            guard val.isFinite else { continue }
            if val < minV { minV = val }
            if val > maxV { maxV = val }
        }
    }
    return (minV, maxV)
}

// MARK: - Infer Command

@available(macOS 14.0, *)
struct Infer: ParsableCommand {
    static var configuration = CommandConfiguration(
        abstract: "Run depth inference on images"
    )
    
    @Option(name: .shortAndLong, help: "Path to DINOv3 backbone CoreML model")
    var backbone: String
    
    @Option(name: .shortAndLong, help: "Path to DualDPT head CoreML model")
    var head: String
    
    @Argument(help: "Input image path(s)")
    var inputs: [String]
    
    @Option(name: .shortAndLong, help: "Output directory")
    var output: String = "./output"
    
    @Option(name: .long, help: "Model size: small, base, large, giant")
    var modelSize: String = "base"
    
    @Option(name: .long, help: "Input image size (default: 518)")
    var inputSize: Int = 518
    
    @Option(name: .long, help: "Colormap for visualization: spectral, turbo, viridis, plasma, magma, grayscale")
    var colormap: String = "spectral"

    @Option(name: .long, help: "Depth visualization style: da3 (inverse-depth, closer=warmer) or depth (raw depth)")
    var depthVizStyle: String = "da3"
    
    @Flag(name: .long, help: "Invert depth visualization (swap near/far colors)")
    var invertDepthViz: Bool = false
    
    @Flag(name: .long, help: "Include ray estimation")
    var includeRays: Bool = false

    @Flag(name: .long, help: "Enable batch processing (uses adaptive batch sizing)")
    var batch: Bool = false

    @Option(name: .long, help: "Override max batch size (default: adaptive)")
    var batchSize: Int?
    
    @Flag(name: .long, help: "Save raw depth as numpy-compatible binary")
    var saveRaw: Bool = false
    
    @Option(name: .long, help: "Output format: da3, npy, raw, png (default: da3)")
    var format: String = "da3"
    
    @Flag(name: .long, help: "Skip PNG visualization (faster, smaller output)")
    var noPng: Bool = false
    
    @Option(name: .long, help: "Memory limit in GB for batching")
    var memoryLimit: Double = 64.0

    @Flag(name: .long, help: "Disable tiled inference (always run a single pass at model resolution, then upscale)")
    var noTiling: Bool = false

    @Flag(name: .long, help: "Force the DualDPT head to run on CPU only (useful for float32 head models when rays are NaN or when GPU execution is unstable)")
    var headCpuOnly: Bool = false

    @Flag(name: .long, help: "Save ray visualization PNGs (direction + confidence) when --include-rays is enabled")
    var rayViz: Bool = false

    @Flag(name: .long, help: "Estimate and print ray-pose intrinsics/extrinsics from predicted rays (debug; requires --include-rays)")
    var rayPose: Bool = false

    @Option(name: .long, help: "Ray-pose subsample factor (debug; default: 16). Increase for very large images.")
    var rayPoseSubsample: Int = 16

    @Option(name: .long, help: "Max tile size in pixels when tiling (default: 1024)")
    var maxTileSize: Int = 1024

    @Option(name: .long, help: "Tile overlap in pixels when tiling (default: 64)")
    var tileOverlap: Int = 64

    @Option(name: .long, help: "Postprocess backend: cpu or metal (default: cpu). Metal accelerates crop/resize + tile blending.")
    var postprocessBackend: String = "cpu"

    @Option(name: .long, help: "Visualization backend for PNGs: cpu or metal (default: cpu). Metal accelerates depth colormap rendering.")
    var vizBackend: String = "cpu"

    @Option(name: .long, help: "Confidence activation for logits heads: linear, expp1, softplus1 (default: linear). Use when you exported `conf_activation=linear`.")
    var confidenceActivation: String = "linear"

    @Option(name: .long, help: "Confidence logit clamp min (used when confidence activation != linear, default: -30)")
    var confidenceLogitClampMin: Float = -30.0

    @Option(name: .long, help: "Confidence logit clamp max (used when confidence activation != linear, default: 30)")
    var confidenceLogitClampMax: Float = 30.0
    
    @Flag(name: .shortAndLong, help: "Verbose output")
    var verbose: Bool = false
    
    func run() throws {
        print("🔮 DA3CoreML - Depth-Anything-3 CoreML")
        print("=====================================")
        
        // Create output directory
        try FileManager.default.createDirectory(atPath: output, withIntermediateDirectories: true)
        
        // Parse model size
        guard let size = DA3CoreML.ModelSize(rawValue: modelSize) else {
            throw ValidationError("Invalid model size: \(modelSize). Use: small, base, large, giant")
        }
        
        // Configure model
        var config = DA3CoreML.Config()
        config.modelSize = size
        config.inputSize = inputSize
        config.memoryLimitGB = memoryLimit
        if let batchSize = batchSize { config.maxBatchSize = batchSize }
        config.enableTiling = !noTiling
        config.maxTileSize = maxTileSize
        config.tileOverlap = tileOverlap
        if headCpuOnly { config.headUseGPU = false }

        if let backend = DA3CoreML.PostprocessBackend(rawValue: postprocessBackend.lowercased()) {
            config.postprocessBackend = backend
        } else {
            throw ValidationError("Invalid --postprocess-backend: \(postprocessBackend). Use: cpu, metal")
        }

        if let act = DA3CoreML.ConfidenceActivation(rawValue: confidenceActivation.lowercased()) {
            config.confidenceActivation = act
        } else {
            throw ValidationError("Invalid --confidence-activation: \(confidenceActivation). Use: linear, expp1, softplus1")
        }
        config.confidenceLogitClampMin = confidenceLogitClampMin
        config.confidenceLogitClampMax = confidenceLogitClampMax
        
        if verbose {
            print("📦 Loading models...")
            print("   Backbone: \(backbone)")
            print("   Head: \(head)")
            print("   Model size: \(size.rawValue)")
            print("   Input size: \(inputSize)x\(inputSize)")
            print("   Tiling: \(noTiling ? "disabled" : "enabled") (maxTileSize=\(maxTileSize), overlap=\(tileOverlap))")
            if headCpuOnly { print("   Head compute: CPU-only (forced)") }
            print("   Postprocess: \(config.postprocessBackend.rawValue)")
            if config.confidenceActivation != .linear {
                print("   Confidence activation: \(config.confidenceActivation.rawValue) (clamp \(config.confidenceLogitClampMin) .. \(config.confidenceLogitClampMax))")
            }
        }
        
        // Load model
        let da3 = try DA3CoreML(backbonePath: backbone, headPath: head, config: config)
        if verbose, da3.config.postprocessBackend != config.postprocessBackend {
            print("   ⚠️ Postprocess backend fell back to \(da3.config.postprocessBackend.rawValue) (Metal unavailable)")
        }
        
        // Parse colormap
        let cmap: Colormap = {
            switch colormap.lowercased() {
            case "spectral": return .spectral
            case "viridis": return .viridis
            case "plasma": return .plasma
            case "magma": return .magma
            case "grayscale", "gray": return .grayscale
            default: return .turbo
            }
        }()

        let vizStyle: DepthVisualizationStyle = {
            switch depthVizStyle.lowercased() {
            case "depth": return .depth
            default: return .da3
            }
        }()
        
        // Choose single-image or batch path
        if batch {
            // Load all images first to let DA3CoreML pick an adaptive batch size
            let loaded: [(String, CGImage)] = inputs.compactMap { path in
                guard let img = loadImage(from: path) else {
                    print("   ⚠️ Failed to load \(path), skipping")
                    return nil
                }
                return (path, img)
            }

            let images = loaded.map { $0.1 }
            if images.isEmpty {
                throw DA3Error.invalidInput("No valid images to process")
            }

            let startBatch = CFAbsoluteTimeGetCurrent()
            let results = try da3.predictBatch(images: images, includeRays: includeRays)
            let batchElapsed = CFAbsoluteTimeGetCurrent() - startBatch
            if verbose {
                print("   ✓ Batch processed \(results.count) images in \(String(format: "%.2f", batchElapsed))s")
            }

            for (idx, result) in results.enumerated() {
                let inputPath = loaded[idx].0
                try saveResult(
                    result,
                    inputPath: inputPath,
                    outputDir: output,
                    cmap: cmap,
                    vizStyle: vizStyle,
                    includeRays: includeRays,
                    format: format,
                    noPng: noPng,
                    rayViz: rayViz
                )
            }
        } else {
            for inputPath in inputs {
                guard let image = loadImage(from: inputPath) else {
                    print("\n📷 Processing: \(inputPath)")
                    print("   ⚠️ Failed to load image, skipping")
                    continue
                }

                print("\n📷 Processing: \(inputPath)")
                if verbose { print("   Image size: \(image.width)x\(image.height)") }

                let startTime = CFAbsoluteTimeGetCurrent()
                let detailed: DA3CoreML.DetailedResult? = rayPose ? try da3.predictDetailed(image: image, includeRays: includeRays) : nil
                let result = try (detailed?.result ?? da3.predict(image: image, includeRays: includeRays))
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                print("   ✓ Inference time: \(String(format: "%.2f", elapsed))s")
                print("   ✓ Depth range: \(String(format: "%.3f", result.minDepth)) - \(String(format: "%.3f", result.maxDepth))")

                if rayPose {
                    if !includeRays {
                        print("   ⚠️ --ray-pose requires --include-rays")
                    } else if let headRays = detailed?.headRays,
                              let headRayConf = detailed?.headRayConfidence {
                        var cfg = DA3RayPoseEstimator.Config()
                        cfg.subsample = max(1, rayPoseSubsample)
                        do {
                            // DA3 convention: ray-pose runs on the **native head ray grid**
                            // (e.g. 296×296), and intrinsics are expressed in the model input
                            // pixel space (typically 518×518).
                            let modelW = detailed?.preprocessInfo?.inputWidth ?? da3.config.inputSize
                            let modelH = detailed?.preprocessInfo?.inputHeight ?? da3.config.inputSize
                            let pose = try DA3RayPoseEstimator.estimatePose(
                                rays: headRays,
                                rayConfidence: headRayConf,
                                imageWidth: modelW,
                                imageHeight: modelH,
                                config: cfg
                            )
                            let K = pose.intrinsics
                            let t = simd_float3(pose.c2w.columns.3.x, pose.c2w.columns.3.y, pose.c2w.columns.3.z)
                            print(String(format: "   ✓ ray-pose K(model): fx=%.3f fy=%.3f cx=%.3f cy=%.3f", K.columns.0.x, K.columns.1.y, K.columns.2.x, K.columns.2.y))
                            print(String(format: "   ✓ ray-pose t: [%.6f %.6f %.6f] (c2w)", t.x, t.y, t.z))

                            if let info = detailed?.preprocessInfo {
                                // Map intrinsics from model-input pixel space back to the original image.
                                //
                                // General mapping:
                                //   x_model = x_orig * scaleX + padLeft
                                //   y_model = y_orig * scaleY + padTop
                                // therefore:
                                //   fx_orig = fx_model / scaleX
                                //   fy_orig = fy_model / scaleY
                                //   cx_orig = (cx_model - padLeft) / scaleX
                                //   cy_orig = (cy_model - padTop)  / scaleY
                                let sx = Float(Double(info.scaledWidth) / Double(image.width))
                                let sy = Float(Double(info.scaledHeight) / Double(image.height))
                                if sx.isFinite, sy.isFinite, sx > 0, sy > 0 {
                                    let fxO = K.columns.0.x / sx
                                    let fyO = K.columns.1.y / sy
                                    let cxO = (K.columns.2.x - Float(info.padLeft)) / sx
                                    let cyO = (K.columns.2.y - Float(info.padTop)) / sy
                                    print(String(format: "   ✓ ray-pose K(orig):  fx=%.3f fy=%.3f cx=%.3f cy=%.3f  (scale=[%.6f,%.6f], pad=[%d,%d])", fxO, fyO, cxO, cyO, sx, sy, info.padLeft, info.padTop))
                                }
                            }
                        } catch {
                            print("   ⚠️ ray-pose estimation failed: \(error)")
                        }
                    } else if let rays = result.rays, let rayConf = result.rayConfidence {
                        // Fallback: postprocessed rays (not DA3 convention). Keep as a debug-only
                        // path for tiled inference where head-resolution rays are unavailable.
                        print("   ⚠️ ray-pose fallback: using postprocessed rays (tiled/upsampled). Results may be unreliable.")
                        var cfg = DA3RayPoseEstimator.Config()
                        cfg.subsample = max(1, rayPoseSubsample)
                        do {
                            let pose = try DA3RayPoseEstimator.estimatePose(
                                rays: rays,
                                rayConfidence: rayConf,
                                imageWidth: image.width,
                                imageHeight: image.height,
                                config: cfg
                            )
                            let K = pose.intrinsics
                            let t = simd_float3(pose.c2w.columns.3.x, pose.c2w.columns.3.y, pose.c2w.columns.3.z)
                            print(String(format: "   ✓ ray-pose K(fallback): fx=%.3f fy=%.3f cx=%.3f cy=%.3f", K.columns.0.x, K.columns.1.y, K.columns.2.x, K.columns.2.y))
                            print(String(format: "   ✓ ray-pose t: [%.6f %.6f %.6f] (c2w)", t.x, t.y, t.z))
                        } catch {
                            print("   ⚠️ ray-pose estimation failed: \(error)")
                        }
                    } else {
                        print("   ⚠️ --ray-pose requested but rays are missing")
                    }
                }

                try saveResult(
                    result,
                    inputPath: inputPath,
                    outputDir: output,
                    cmap: cmap,
                    vizStyle: vizStyle,
                    includeRays: includeRays,
                    format: format,
                    noPng: noPng,
                    rayViz: rayViz
                )
            }
        }
        
        print("\n✅ Done!")
    }
    
    func loadImage(from path: String) -> CGImage? {
        return CLIImageLoader.loadImage(from: path)
    }
    
    func saveImage(_ image: CGImage, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw DA3Error.imageProcessingFailed("Failed to create image destination")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw DA3Error.imageProcessingFailed("Failed to write image")
        }
    }
    
    func saveRawDepth(_ array: MLMultiArray, to path: String) throws {
        let values: [Float] = (try? MLMultiArrayFloatReader(array))?.readAll()
            ?? (0..<array.count).map { array[$0].floatValue }
        var data = Data()
        data.reserveCapacity(values.count * MemoryLayout<Float>.size)
        for v in values {
            var value = v
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        try data.write(to: URL(fileURLWithPath: path))
    }
    
    func saveRawArray(_ array: MLMultiArray, to path: String) throws {
        var data = Data()
        // Write shape
        var shapeCount = UInt32(array.shape.count)
        withUnsafeBytes(of: &shapeCount) { data.append(contentsOf: $0) }
        for dim in array.shape {
            var dimVal = UInt32(dim.intValue)
            withUnsafeBytes(of: &dimVal) { data.append(contentsOf: $0) }
        }
        // Write data
        let values: [Float] = (try? MLMultiArrayFloatReader(array))?.readAll()
            ?? (0..<array.count).map { array[$0].floatValue }
        data.reserveCapacity(data.count + values.count * MemoryLayout<Float>.size)
        for v in values {
            var value = v
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        try data.write(to: URL(fileURLWithPath: path))
    }

    /// Common save routine used by single and batch flows
    func saveResult(
        _ result: DA3CoreML.Result,
        inputPath: String,
        outputDir: String,
        cmap: Colormap,
        vizStyle: DepthVisualizationStyle,
        includeRays: Bool,
        format: String,
        noPng: Bool,
        rayViz: Bool
    ) throws {
        let baseName = (inputPath as NSString).deletingPathExtension
        let fileName = (baseName as NSString).lastPathComponent

        var writerConfig = DA3OutputWriter.Config()
        writerConfig.includeRays = includeRays
        writerConfig.colormap = cmap
        writerConfig.depthVisualizationStyle = vizStyle
        writerConfig.invertDepthVisualization = invertDepthViz
        writerConfig.compress = true
        if let vb = DA3OutputWriter.Config.VisualizationBackend(rawValue: vizBackend.lowercased()) {
            writerConfig.visualizationBackend = vb
        }

        let writer = DA3OutputWriter(config: writerConfig)
        let imageInfo = ImageInfo(sourcePath: inputPath)

        let outputFormat: DA3OutputWriter.OutputFormat
        switch format.lowercased() {
        case "npy": outputFormat = .npy
        case "raw": outputFormat = .raw
        case "png": outputFormat = .png
        default: outputFormat = .da3
        }

        let outputPath = "\(outputDir)/\(fileName)"
        try writer.save(result, to: outputPath, format: outputFormat, imageInfo: imageInfo)
        print("   ✓ Saved: \(outputPath).\(format)")

        if !noPng && outputFormat != .png {
            try writer.save(result, to: outputPath, format: .png)
            print("   ✓ Saved visualization: \(outputPath)_depth.png")
        }

        // Optional ray visualizations (small sanity-check outputs; avoids writing huge ray tensors as .npy)
        if rayViz && includeRays && !noPng {
            guard let rays = result.rays, let rayConf = result.rayConfidence else {
                print("   ⚠️ Ray viz requested but rays are missing")
                return
            }

            if let dirImg = makeRayDirectionImage(rays) {
                let p = "\(outputPath)_rays_dir.png"
                try saveImage(dirImg, to: p)
                print("   ✓ Saved ray dir viz: \(p)")
            } else {
                print("   ⚠️ Failed to create ray direction visualization")
            }

            if let confImg = makeRayConfidenceImage(rayConf) {
                let p = "\(outputPath)_rays_conf.png"
                try saveImage(confImg, to: p)
                print("   ✓ Saved ray conf viz: \(p)")
            } else {
                print("   ⚠️ Failed to create ray confidence visualization")
            }
        }
    }

    private func makeRayDirectionImage(_ rays: MLMultiArray) -> CGImage? {
        guard let reader = try? MLMultiArrayFloatReader(rays) else { return nil }
        let shape = reader.shape
        let (c, h, w): (Int, Int, Int) = {
            switch shape.count {
            case 4: return (shape[1], shape[2], shape[3]) // [B, C, H, W]
            case 3: return (shape[0], shape[1], shape[2]) // [C, H, W]
            default: return (0, 0, 0)
            }
        }()
        guard c >= 3, h > 0, w > 0 else { return nil }

        // Render only the first 3 ray channels as normalized direction (dx,dy,dz) -> RGB.
        let hw = h * w
        var rgba = [UInt8](repeating: 0, count: hw * 4)

        func ray(_ ch: Int, _ y: Int, _ x: Int) -> Float {
            switch shape.count {
            case 4:
                return reader.read(0, ch, y, x)
            case 3:
                return reader.read(ch, y, x)
            default:
                return 0
            }
        }

        for y in 0..<h {
            for x in 0..<w {
                let dx = ray(0, y, x)
                let dy = ray(1, y, x)
                let dz = ray(2, y, x)
                let len = max(1e-8, sqrt(dx * dx + dy * dy + dz * dz))
                let r = min(1, max(0, 0.5 * (dx / len) + 0.5))
                let g = min(1, max(0, 0.5 * (dy / len) + 0.5))
                let b = min(1, max(0, 0.5 * (dz / len) + 0.5))
                let i = y * w + x
                let o = i * 4
                rgba[o] = UInt8(r * 255)
                rgba[o + 1] = UInt8(g * 255)
                rgba[o + 2] = UInt8(b * 255)
                rgba[o + 3] = 255
            }
        }

        return makeCGImageRGBA(width: w, height: h, rgba: rgba)
    }

    private func makeRayConfidenceImage(_ rayConf: MLMultiArray) -> CGImage? {
        guard let reader = try? MLMultiArrayFloatReader(rayConf) else { return nil }
        let shape = reader.shape
        let (h, w): (Int, Int) = {
            switch shape.count {
            case 4: return (shape[2], shape[3]) // [B, C, H, W]
            case 3: return (shape[1], shape[2]) // [C, H, W] or [B, H, W]
            case 2: return (shape[0], shape[1]) // [H, W]
            default: return (0, 0)
            }
        }()
        guard h > 0, w > 0 else { return nil }

        let hw = h * w

        func confAt(_ y: Int, _ x: Int) -> Float {
            switch shape.count {
            case 4:
                return reader.read(0, 0, y, x)
            case 3:
                // Treat as [C,H,W] and read channel 0.
                return reader.read(0, y, x)
            case 2:
                return reader.read(y, x)
            default:
                return 0
            }
        }

        // Compute log-space percentiles for visualization (robust to huge expp1 ranges).
        let maxSamples = 1_000_000
        let step = max(1, hw / maxSamples)
        var samples: [Float] = []
        samples.reserveCapacity(min(maxSamples, hw))

        var i = 0
        while i < hw {
            let y = i / w
            let x = i - y * w
            let v = confAt(y, x)
            if v.isFinite, v > 0 {
                samples.append(log(v))
            }
            i += step
        }

        samples.sort()
        let lo = samples.isEmpty ? 0 : samples[Int(Float(samples.count - 1) * 0.02)]
        let hi = samples.isEmpty ? 1 : samples[Int(Float(samples.count - 1) * 0.98)]
        let denom = max(1e-6, hi - lo)

        var rgba = [UInt8](repeating: 0, count: hw * 4)
        for y in 0..<h {
            for x in 0..<w {
                let v = confAt(y, x)
                let t: Float
                if v.isFinite, v > 0 {
                    t = min(1, max(0, (log(v) - lo) / denom))
                } else {
                    t = 0
                }
                let g = UInt8(t * 255)
                let i = y * w + x
                let o = i * 4
                rgba[o] = g
                rgba[o + 1] = g
                rgba[o + 2] = g
                rgba[o + 3] = 255
            }
        }

        return makeCGImageRGBA(width: w, height: h, rgba: rgba)
    }

    private func makeCGImageRGBA(width: Int, height: Int, rgba: [UInt8]) -> CGImage? {
        guard rgba.count == width * height * 4 else { return nil }
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        return rgba.withUnsafeBytes { rawBuf in
            guard let base = rawBuf.baseAddress,
                  let provider = CGDataProvider(data: Data(bytes: base, count: rgba.count) as CFData) else {
                return nil
            }
            return CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: cs,
                bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
            )
        }
    }
}

// MARK: - Convert Command

@available(macOS 14.0, *)
struct Convert: ParsableCommand {
    static var configuration = CommandConfiguration(
        abstract: "Convert PyTorch models to CoreML (requires Python)"
    )
    
    @Option(name: .shortAndLong, help: "Model to convert: dinov3, dualdpt, or all")
    var model: String = "all"
    
    @Option(name: .shortAndLong, help: "Model size: small, base, large, giant")
    var size: String = "base"
    
    @Option(name: .shortAndLong, help: "Output directory for CoreML models")
    var output: String = "./Models"
    
    @Option(name: .long, help: "HuggingFace model name for DINOv3")
    var hfModel: String?
    
    @Option(name: .long, help: "Path to DA3 checkpoint for DualDPT")
    var checkpoint: String?
    
    @Flag(name: .shortAndLong, help: "Verbose output")
    var verbose: Bool = false
    
    func run() throws {
        print("🔄 DA3CoreML Model Converter")
        print("============================")
        
        // Create output directory
        try FileManager.default.createDirectory(atPath: output, withIntermediateDirectories: true)
        
        let scriptsDir = getScriptsDirectory()
        
        if model == "dinov3" || model == "all" {
            print("\n📦 Converting DINOv3...")
            try convertDINOv3(scriptsDir: scriptsDir)
        }
        
        if model == "dualdpt" || model == "all" {
            print("\n📦 Converting DualDPT...")
            try convertDualDPT(scriptsDir: scriptsDir)
        }
        
        print("\n✅ Conversion complete!")
        print("   Models saved to: \(output)")
    }
    
    func getScriptsDirectory() -> String {
        // Get path relative to executable
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        return "\(execDir)/../Scripts"
    }
    
    func convertDINOv3(scriptsDir: String) throws {
        let hfModelName = hfModel ?? "facebook/dinov2-\(size)"
        let outputPath = "\(output)/dinov3_\(size).mlpackage"
        
        print("   Model: \(hfModelName)")
        print("   Output: \(outputPath)")
        
        let script = "\(scriptsDir)/convert_dinov3_to_coreml.py"
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [
            script,
            "--model", hfModelName,
            "--output", outputPath,
            "--precision", "float16"
        ]
        
        if verbose {
            process.standardOutput = FileHandle.standardOutput
            process.standardError = FileHandle.standardError
        }
        
        try process.run()
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            throw DA3Error.conversionFailed("DINOv3 conversion failed with status \(process.terminationStatus)")
        }
    }
    
    func convertDualDPT(scriptsDir: String) throws {
        let outputPath = "\(output)/dualdpt_\(size).mlpackage"
        
        print("   Output: \(outputPath)")
        
        if let ckpt = checkpoint {
            print("   Checkpoint: \(ckpt)")
        } else {
            print("   ⚠️ No checkpoint provided - will need manual conversion")
            print("   Use: --checkpoint <path_to_da3_checkpoint.pth>")
            return
        }
        
        let script = "\(scriptsDir)/convert_dualdpt_to_coreml.py"
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [
            script,
            "--checkpoint", checkpoint!,
            "--output", outputPath,
            "--size", size,
            "--precision", "float16"
        ]
        
        if verbose {
            process.standardOutput = FileHandle.standardOutput
            process.standardError = FileHandle.standardError
        }
        
        try process.run()
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            throw DA3Error.conversionFailed("DualDPT conversion failed with status \(process.terminationStatus)")
        }
    }
}

// MARK: - Benchmark Command

@available(macOS 14.0, *)
struct Benchmark: ParsableCommand {
    static var configuration = CommandConfiguration(
        abstract: "Benchmark inference performance"
    )
    
    @Option(name: .shortAndLong, help: "Path to DINOv3 backbone CoreML model")
    var backbone: String
    
    @Option(name: .shortAndLong, help: "Path to DualDPT head CoreML model")
    var head: String
    
    @Option(name: .long, help: "Model size: small, base, large, giant")
    var modelSize: String = "base"
    
    @Option(name: .long, help: "Number of warmup iterations")
    var warmup: Int = 3
    
    @Option(name: .long, help: "Number of benchmark iterations")
    var iterations: Int = 10
    
    @Option(name: .long, help: "Input image sizes to test (comma-separated)")
    var sizes: String = "256,518,768,1024"
    
    func run() throws {
        print("⚡ DA3CoreML Benchmark")
        print("======================")
        
        guard let size = DA3CoreML.ModelSize(rawValue: modelSize) else {
            throw ValidationError("Invalid model size: \(modelSize)")
        }
        
        var config = DA3CoreML.Config()
        config.modelSize = size
        
        print("\nLoading model...")
        let da3 = try DA3CoreML(backbonePath: backbone, headPath: head, config: config)
        
        let testSizes = sizes.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        
        print("\nBenchmarking with \(iterations) iterations (+ \(warmup) warmup)...")
        print("─────────────────────────────────────────────")
        print("Size      │ Avg (ms) │ Min (ms) │ Max (ms)")
        print("─────────────────────────────────────────────")
        
        for testSize in testSizes {
            // Create test image
            let testImage = createTestImage(width: testSize, height: testSize)
            
            // Warmup
            for _ in 0..<warmup {
                _ = try? da3.predict(image: testImage, includeRays: false)
            }
            
            // Benchmark
            var times: [Double] = []
            for _ in 0..<iterations {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try? da3.predict(image: testImage, includeRays: false)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                times.append(elapsed)
            }
            
            let avg = times.reduce(0, +) / Double(times.count)
            let minTime = times.min() ?? 0
            let maxTime = times.max() ?? 0
            
            print(String(format: "%4dx%-4d │ %8.1f │ %8.1f │ %8.1f", testSize, testSize, avg, minTime, maxTime))
        }
        
        print("─────────────────────────────────────────────")
    }
    
    func createTestImage(width: Int, height: Int) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        
        // Fill with gradient
        for y in 0..<height {
            for x in 0..<width {
                let r = UInt8(x * 255 / width)
                let g = UInt8(y * 255 / height)
                let b = UInt8(128)
                context.setFillColor(red: CGFloat(r)/255, green: CGFloat(g)/255, blue: CGFloat(b)/255, alpha: 1)
                context.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        
        return context.makeImage()!
    }
}

// MARK: - To3DGS Command

@available(macOS 14.0, *)
struct To3DGS: ParsableCommand {
    static var configuration = CommandConfiguration(
        abstract: "Convert DA3 depth files to 3D Gaussian Splatting PLY"
    )
    
    @Argument(help: "Input .da3 file(s) or directory")
    var inputs: [String]
    
    @Option(name: .shortAndLong, help: "Output directory for PLY files")
    var output: String = "./output_3dgs"
    
    @Option(name: .long, help: "Source image for colors (optional)")
    var sourceImage: String?
    
    @Option(name: .long, help: "Subsample factor (1=all pixels, 2=every 2nd, etc.)")
    var subsample: Int = 2
    
    @Option(name: .long, help: "Minimum confidence threshold [0-1]")
    var minConfidence: Float = 0.3
    
    @Option(name: .long, help: "Gaussian scale")
    var gaussianScale: Float = 0.01
    
    @Option(name: .long, help: "Field of view in degrees (for camera estimation)")
    var fov: Float = 50
    
    @Flag(name: .long, help: "Output ASCII PLY instead of binary")
    var ascii: Bool = false
    
    @Flag(name: .shortAndLong, help: "Verbose output")
    var verbose: Bool = false
    
    func run() throws {
        print("🔮 DA3CoreML - Depth to 3D Gaussian Splatting")
        print("==============================================")
        print("Note: `to3-dgs` is a simple depth→point→Gaussian initializer for debugging. For DA3-quality feed-forward splats and multi-view fusion, use `da3-coreml fuse --gshead ...`.")
        
        try FileManager.default.createDirectory(atPath: output, withIntermediateDirectories: true)
        
        // Configure converter
        var config = DA3DepthTo3DGS.Config()
        config.subsample = subsample
        config.minConfidence = minConfidence
        config.gaussianScale = gaussianScale
        
        let converter = DA3DepthTo3DGS(config: config)
        let reader = DA3OutputReader()
        let writer = DA3PLYWriter()
        
        // Load source image if provided
        var sourceImg: CGImage?
        if let imgPath = sourceImage {
            sourceImg = loadImage(from: imgPath)
        }
        
        // Process each input
        for inputPath in inputs {
            let url = URL(fileURLWithPath: inputPath)
            
            if url.pathExtension == "da3" {
                try processDA3File(inputPath, converter: converter, reader: reader, writer: writer, sourceImg: sourceImg)
            } else {
                // Try to find .da3 files in directory
                let files = try FileManager.default.contentsOfDirectory(atPath: inputPath)
                for file in files where file.hasSuffix(".da3") {
                    let fullPath = "\(inputPath)/\(file)"
                    try processDA3File(fullPath, converter: converter, reader: reader, writer: writer, sourceImg: sourceImg)
                }
            }
        }
        
        print("\n✅ Done! PLY files saved to: \(output)")
    }
    
    func processDA3File(
        _ path: String,
        converter: DA3DepthTo3DGS,
        reader: DA3OutputReader,
        writer: DA3PLYWriter,
        sourceImg: CGImage?
    ) throws {
        print("\n📄 Processing: \(path)")
        
        // Load depth data
        let data = try reader.load(from: path)
        
        if verbose {
            print("   Dimensions: \(data.width)x\(data.height)")
            print("   Depth range: \(data.depthMin) - \(data.depthMax)")
        }
        
        // Convert to Gaussians
        let cloud = try converter.convert(data: data, sourceImage: sourceImg)
        
        print("   ✓ Generated \(cloud.count) Gaussians")
        
        // Write PLY
        let baseName = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        let plyPath = "\(output)/\(baseName).ply"
        
        try writer.write(
            cloud,
            to: plyPath,
            format: ascii ? .ascii : .binary,
            comments: [
                "mode: depth-only-init (to3-dgs)",
                "warning: no gshead; rotations/scale are heuristic",
            ]
        )
        print("   ✓ Saved: \(plyPath)")
    }
    
    func loadImage(from path: String) -> CGImage? {
        return CLIImageLoader.loadImage(from: path)
    }
}
