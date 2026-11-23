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
    struct Artifact: Sendable {
        let name: String
        let repo: String
        let revision: String
        let directoryName: String
        let requiresAlignedWeights: Bool
        var excludedFile: String? = nil

        static func named(_ name: String) -> Self? {
            if name == "inkling-2-mini-mei-q4" {
                return Self(name: name, repo: "Tostibrown/Inkling-2-Mini-Mei-Vocab-Q4",
                            revision: "ab7a0a95072f7822cd47ee0eba93f7d7c1f770a7",
                            directoryName: name, requiresAlignedWeights: false)
            }
            if name == "inkling-2-mini" {
                return Self(name: name, repo: "nativ-community/Inkling-2-Mini-nvfp4",
                            revision: "26f71eef259a1dd7fc476778d67020351e0dfb8c",
                            directoryName: name, requiresAlignedWeights: false,
                            excludedFile: "mtp.safetensors")
            }
            return ModelTuningRegistry.named(name).map(fromTuning)
        }

        static func fromTuning(_ tuning: ModelTuning) -> Self {
            Self(name: tuning.name, repo: tuning.repo, revision: tuning.revision,
                 directoryName: tuning.repo.split(separator: "/").last.map(String.init) ?? tuning.name,
                 requiresAlignedWeights: tuning.requiresAlignedWeights)
        }

        func destination(in root: URL) -> String {
            root.appendingPathComponent(directoryName, isDirectory: true).path
        }
    }

    static var modelCache: URL {
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".cache/mei/models", isDirectory: true)
    }

    public static var names: [String] { ModelTuningRegistry.names + ["inkling-2-mini", "inkling-2-mini-mei-q4"] }

    public struct Plan: Sendable {
        let artifact: Artifact
        public let destination: String

        init(artifact: Artifact, destination: String) {
            self.artifact = artifact
            self.destination = destination
        }

        init(tuning: ModelTuning, destination: String) {
            self.init(artifact: .fromTuning(tuning), destination: destination)
        }

        var arguments: [String] {
            var args = ["download", artifact.repo, "--revision", artifact.revision]
            if let excludedFile = artifact.excludedFile {
                args += ["--exclude", excludedFile]
            }
            return args + ["--local-dir", destination]
        }
    }

    public static func defaultDestination(for tuning: ModelTuning) -> String {
        Artifact.fromTuning(tuning).destination(in: modelCache)
    }

    /// Display the same arguments passed to the Hugging Face CLI.
    public static func downloadCommand(_ plan: Plan) -> String {
        let args = plan.arguments.map { arg in
            if arg.range(of: #"^[A-Za-z0-9_./-]+$"#, options: .regularExpression) != nil {
                return arg
            }
            return "'" + arg.replacingOccurrences(of: "'", with: "'\\''") + "'"
        }
        return (["hf"] + args).joined(separator: " ")
    }

    public static func run(profileName: String, destination: String?,
                           dryRun: Bool = false) -> Int32 {
        guard let artifact = Artifact.named(profileName) else {
            FileHandle.standardError.write(Data((
                "mei: '\(profileName)' is not a supported model. Known models: "
                + names.joined(separator: ", ") + "\n").utf8))
            return 2
        }
        let plan = Plan(artifact: artifact,
                        destination: destination ?? artifact.destination(in: modelCache))
        print("mei: \(artifact.name)")
        print("mei: repo \(artifact.repo) @ \(artifact.revision)")
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

        let status = spawn(hf, plan.arguments)
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
            if !plan.artifact.requiresAlignedWeights {
                print("mei: downloaded — alignment is optional for this profile (\(bad)/\(total) unaligned shards)")
                return 0
            }
            FileHandle.standardError.write(Data((
                "mei: WARNING \(bad) of \(total) shards are not naturally "
                + "aligned. This profile's settings were measured on an aligned "
                + "artifact; serving this one costs memory and long-context "
                + "prefill speed.\n").utf8))
            return 4
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
