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
