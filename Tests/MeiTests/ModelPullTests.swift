import Foundation
import XCTest
@testable import MeiCore

final class ModelPullTests: XCTestCase {
    private func tuning(_ name: String) throws -> ModelTuning {
        try XCTUnwrap(ModelTuningRegistry.named(name))
    }

    func testPullTargetsTheProfilesPinnedRepoAndRevision() throws {
        let t = try tuning("ornith-1.5-35b-a3b")
        let plan = ModelPull.Plan(tuning: t, destination: "/tmp/x")
        let cmd = ModelPull.downloadCommand(plan)
        // The whole point of pull is fetching the artifact the settings were
        // measured on, so the pinned revision must appear in the command.
        XCTAssertTrue(cmd.contains(t.repo), cmd)
        XCTAssertTrue(cmd.contains(t.revision), cmd)
        XCTAssertTrue(cmd.contains("/tmp/x"), cmd)
        XCTAssertFalse(cmd.contains("main"), "must pin a revision, not a branch")
    }

    func testOrnithPullsTheAlignedRepackNotTheOfficialCheckpoint() throws {
        // Pulling the official checkpoint costs 4.7 GB and 3.2x on an 80k
        // prefill, invisibly. If this ever points back at it, that regression
        // ships silently.
        let t = try tuning("ornith-1.5-35b-a3b")
        XCTAssertTrue(t.repo.contains("aligned"), t.repo)
        XCTAssertTrue(t.requiresAlignedWeights)
    }

    func testDefaultDestinationIsUserLocalAndNamedAfterTheRepo() throws {
        let t = try tuning("qwen3.6-35b-a3b-text")
        let dest = ModelPull.defaultDestination(for: t)
        XCTAssertTrue(dest.hasPrefix(NSHomeDirectory()), dest)
        XCTAssertTrue(dest.hasSuffix("Qwen3.6-35B-A3B-4bit-textonly"), dest)
    }

    func testUnknownProfileFailsWithoutTouchingTheNetwork() {
        XCTAssertEqual(ModelPull.run(profileName: "not-a-model",
                                     destination: "/tmp/should-not-appear"), 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: "/tmp/should-not-appear"))
    }

    func testDryRunDownloadsNothing() throws {
        let dest = NSTemporaryDirectory() + "mei-pull-dryrun-\(UUID().uuidString)"
        XCTAssertEqual(ModelPull.run(profileName: "ornith-1.5-35b-a3b",
                                     destination: dest, dryRun: true), 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dest),
                       "a dry run must not create the destination")
    }
}

final class ModelArtifactCheckTests: XCTestCase {
    /// safetensors: 8-byte little-endian header length, then the JSON header,
    /// then the data segment. Alignment is a property of `8 + headerLength`.
    private func makeShard(inDirectory dir: URL, headerLength: Int) throws {
        let header = Data(String(repeating: " ", count: headerLength).utf8)
        var out = Data()
        withUnsafeBytes(of: UInt64(headerLength).littleEndian) { out.append(contentsOf: $0) }
        out.append(header)
        out.append(Data([0, 1, 2, 3]))
        try out.write(to: dir.appendingPathComponent("model-00001-of-00001.safetensors"))
    }

    func testAlignedShardIsRecognised() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("align-ok-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try makeShard(inDirectory: dir, headerLength: 56)   // 8 + 56 = 64
        XCTAssertEqual(ModelArtifactCheck.alignment(ofModelDirectory: dir.path), .aligned)
    }

    func testUnalignedShardIsCaught() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("align-bad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try makeShard(inDirectory: dir, headerLength: 53)   // 8 + 53 = 61, not 8-aligned
        XCTAssertEqual(ModelArtifactCheck.alignment(ofModelDirectory: dir.path),
                       .unaligned(unalignedShards: 1, totalShards: 1))
    }

    func testManifestShortCircuitsTheCheck() throws {
        // Our repack ships a manifest; trust it rather than re-reading shards.
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("align-man-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{}".utf8).write(to: dir.appendingPathComponent("MEI_ALIGN_MANIFEST.json"))
        XCTAssertEqual(ModelArtifactCheck.alignment(ofModelDirectory: dir.path), .aligned)
    }

    func testEmptyDirectoryIsUnknownNotAligned() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("align-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        if case .unknown = ModelArtifactCheck.alignment(ofModelDirectory: dir.path) {} else {
            XCTFail("an empty directory must not report as aligned")
        }
    }
}
