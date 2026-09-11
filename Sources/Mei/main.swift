import Foundation
import MeiCore
import MLX

@main
struct MeiMain {
    static func main() async {
        if CommandLine.arguments.dropFirst().contains("--version") {
            print("mei \(ServerConfig.version)")
            return
        }
        let config: ServerConfig
        do {
            config = try ServerConfig.parse()
        } catch {
            FileHandle.standardError.write(Data("mei: \(error.localizedDescription)\n\n\(ServerConfig.usage)\n".utf8))
            exit(2)
        }

        config.optimizationProfile.applyRuntimeEnvironment(
            force: config.requestedOptimizationProfile == .ornith)
        print("mei: optimization profile \(config.optimizationProfile.rawValue) (requested \(config.requestedOptimizationProfile.rawValue), prefill \(config.prefillStepSize), compiled-decode \(config.enableCompiledDecode), kv-window \(config.maxKVWindowSize), ssm-anchors \(config.ssmAnchorBoundaryCount))")
        // If the operator sets no flags, the profile is choosing for them — so
        // say what it chose. Silent auto-tuning is only an improvement if it is
        // visible when something looks wrong.
        if config.optimizationProfile.isOrnith {
            let compile = ProcessInfo.processInfo.environment["VMLX_ENABLE_UNSAFE_COMPILE"] ?? "unset"
            let fused = ProcessInfo.processInfo.environment["VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES"] ?? "unset"
            print("mei: profile applied max-tokens \(config.maxTokensDefault)"
                + ", unsafe-compile \(compile), fused-gate-up-limit \(fused)")
        }
        fflush(stdout)
        if config.servedModelIDWasDefaulted {
            print("mei: no --served-model-id given; serving as "
                + "\(config.servedModelID) (clients must use this exact id)")
            fflush(stdout)
        }
        if config.autoEnabledAnchorKVCacheDir {
            // Say it out loud: the operator asked for anchors, not for a cache
            // directory, and without one the feature would have done nothing.
            print("mei: --ssm-anchor-boundaries needs a durable KV tier; "
                + "using disposable cache at \(config.kvCacheDir) "
                + "(pass --kv-cache-dir to keep it across restarts)")
            fflush(stdout)
        }
        print("mei: loading model from \(config.modelDirectory) (served id: \(config.servedModelID))...")
        fflush(stdout)
        let engine: Engine
        do {
            engine = try await Engine.load(config: config)
        } catch {
            FileHandle.standardError.write(Data("mei: model load failed: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
        print("mei: model loaded")
        fflush(stdout)
        let device = GPU.deviceInfo()
        let loaded = await engine.loadMemory
        print("mei: device \(device.architecture) memory \(device.memorySize) bytes; recommended working set \(GPU.maxRecommendedWorkingSetBytes() ?? -1) bytes")
        print("mei: memory after load: active \(loaded?.activeMemory ?? -1) cache \(loaded?.cacheMemory ?? -1) peak \(loaded?.peakMemory ?? -1) bytes; limit \(Memory.memoryLimit) cache-limit \(Memory.cacheLimit)")
        fflush(stdout)

        let router = Router(engine: engine, config: config)
        let server: MeiHTTPServer
        do {
            server = try MeiHTTPServer(router: router, config: config)
        } catch {
            FileHandle.standardError.write(Data("mei: failed to bind \(config.host):\(config.port): \(error.localizedDescription)\n".utf8))
            exit(1)
        }
        print("mei: listening on http://\(config.host):\(config.port) (context cap \(config.contextCap), prefill step \(config.prefillStepSize), kv-bits \(config.kvBits.map(String.init) ?? "none"))")
        fflush(stdout)

        do {
            try await server.run()
        } catch {
            FileHandle.standardError.write(Data("mei: server error: \(error.localizedDescription)\n".utf8))
            server.shutdown()
            exit(1)
        }
    }
}