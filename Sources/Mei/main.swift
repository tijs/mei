import Foundation
import MeiCore

@main
struct MeiMain {
    static func main() async {
        let config: ServerConfig
        do {
            config = try ServerConfig.parse()
        } catch {
            FileHandle.standardError.write(Data("mei: \(error.localizedDescription)\n\n\(ServerConfig.usage)\n".utf8))
            exit(2)
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