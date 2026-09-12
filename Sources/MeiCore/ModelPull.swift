import Foundation

/// `mei pull <profile>` — fetch the exact artifact a profile's settings were
/// measured on, then verify it.
///
/// This exists because the gap between "downloaded a model" and "serving it
/// well" is invisible. Ornith's profile pins an aligned repack; pull the
/// official checkpoint instead and you get a server 4.7 GB fatter and 3.2x
/// slower on an 80k prefill, with no symptom to search for. Naming the model
/// should get you the right bytes, not just the right flags.
///
/// The download itself is delegated to the `hf` CLI rather than reimplemented.
/// HuggingFace's transfer path involves LFS, content-addressed dedup and
/// resumable chunking; a hand-rolled URLSession downloader would be a worse
/// version of a tool the install guide already requires.
public enum ModelPull {
    public struct Plan: Sendable {
        public let tuning: ModelTuning
        public let destination: String
    }

    public static func defaultDestination(for tuning: ModelTuning) -> String {
        let base = (NSHomeDirectory() as NSString)
            .appendingPathComponent(".cache/mei/models")
        let leaf = tuning.repo.split(separator: "/").last.map(String.init) ?? tuning.name
        return (base as NSString).appendingPathComponent(leaf)
    }

    /// The command a user could run by hand. Printed on every path, so the
    /// operation is never a black box.
    public static func downloadCommand(_ plan: Plan) -> String {
        "hf download \(plan.tuning.repo) --revision \(plan.tuning.revision) "
            + "--local-dir \(plan.destination)"
    }

    public static func run(profileName: String, destination: String?,
                           dryRun: Bool = false) -> Int32 {
        guard let tuning = ModelTuningRegistry.named(profileName) else {
            FileHandle.standardError.write(Data((
                "mei: '\(profileName)' is not a supported model. Known profiles: "
                + ModelTuningRegistry.names.joined(separator: ", ") + "\n").utf8))
            return 2
        }
        let plan = Plan(tuning: tuning,
                        destination: destination ?? defaultDestination(for: tuning))
        print("mei: \(tuning.name) — \(tuning.model)")
        print("mei: repo \(tuning.repo) @ \(tuning.revision)")
        print("mei: destination \(plan.destination)")
        print("mei: $ \(downloadCommand(plan))")

        if dryRun {
            // Inspecting the plan must not cost a 19 GB download. Found by
            // running `mei pull` to check its output and watching it start one.
            print("mei: dry run — nothing downloaded")
            switch ModelArtifactCheck.alignment(ofModelDirectory: plan.destination) {
            case .aligned: print("mei: destination already holds aligned weights")
            case .unaligned(let bad, let total):
                print("mei: destination holds \(bad)/\(total) unaligned shards")
            case .unknown: print("mei: destination is empty or unreadable")
            }
            return 0
        }
        guard let hf = which("hf") else {
            FileHandle.standardError.write(Data((
                "mei: the `hf` CLI is not on PATH. Install it with "
                + "`uv tool install --upgrade huggingface_hub`, or run the "
                + "command above yourself.\n").utf8))
            return 3
        }

        let status = spawn(hf, [
            "download", tuning.repo,
            "--revision", tuning.revision,
            "--local-dir", plan.destination,
        ])
        guard status == 0 else {
            FileHandle.standardError.write(Data(
                "mei: download failed (hf exited \(status))\n".utf8))
            return status
        }
        return verify(plan)
    }

    /// Downloading the right repo is not the same as having the right bytes.
    private static func verify(_ plan: Plan) -> Int32 {
        switch ModelArtifactCheck.alignment(ofModelDirectory: plan.destination) {
        case .aligned:
            print("mei: verified — tensors are naturally aligned")
            return 0
        case .unaligned(let bad, let total):
            FileHandle.standardError.write(Data((
                "mei: WARNING \(bad) of \(total) shards are not naturally "
                + "aligned. This profile's settings were measured on an aligned "
                + "artifact; serving this one costs memory and long-context "
                + "prefill speed.\n").utf8))
            return plan.tuning.requiresAlignedWeights ? 4 : 0
        case .unknown(let why):
            FileHandle.standardError.write(Data(
                "mei: could not verify alignment (\(why))\n".utf8))
            return 0   // do not fail a download over an unreadable check
        }
    }

    private static func which(_ tool: String) -> String? {
        let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
        for dir in path.split(separator: ":") {
            let candidate = (String(dir) as NSString).appendingPathComponent(tool)
            if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
        }
        return nil
    }

    private static func spawn(_ launchPath: String, _ args: [String]) -> Int32 {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launchPath)
        p.arguments = args
        do { try p.run() } catch { return 127 }
        p.waitUntilExit()
        return p.terminationStatus
    }
}
