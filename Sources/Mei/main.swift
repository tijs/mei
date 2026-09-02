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
        fflush(stdout)
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