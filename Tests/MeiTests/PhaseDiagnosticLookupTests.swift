import Foundation
import MLXLMCommon
import XCTest

final class PhaseDiagnosticLookupTests: XCTestCase {
    func testLiveFlagsRetainExactOneAndLegacyORSemantics() {
        let current = "VMLX_NATIVE_MTP_PHASE_DIAG"
        let legacy = "VMLINUX_NATIVE_MTP_PHASE_DIAG"
        let saved = ProcessInfo.processInfo.environment
        func assign(_ key: String, _ value: String?) {
            if let value { setenv(key, value, 1) } else { unsetenv(key) }
        }
        defer {
            assign(current, saved[current])
            assign(legacy, saved[legacy])
            NativeMTPPhaseDiagnostics.reset()
        }
        let values: [String?] = [nil, "", "0", "1", "true", "yes", "01"]
        for primary in values {
            for fallback in values {
                assign(current, primary)
                assign(legacy, fallback)
                XCTAssertEqual(NativeMTPPhaseDiagnostics.enabled, primary == "1" || fallback == "1")
            }
        }
        NativeMTPPhaseDiagnostics.reset()
        assign(current, "1")
        assign(legacy, nil)
        NativeMTPPhaseDiagnostics.record("fixture", seconds: 0.25)
        assign(current, "0")
        NativeMTPPhaseDiagnostics.record("fixture", seconds: 9)
        let snapshot = NativeMTPPhaseDiagnostics.snapshot()
        XCTAssertEqual(snapshot.calls, ["fixture": 1])
        XCTAssertEqual(snapshot.seconds, ["fixture": 0.25])
    }
}
