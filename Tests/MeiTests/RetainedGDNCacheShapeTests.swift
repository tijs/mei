import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM
import MLXLMCommon

final class RetainedGDNCacheShapeTests: XCTestCase {
    func testMalformedRestoredRecurrenceMatchesFreshState() throws {
        // Retained geometry with a tiny hidden projection; no checkpoint/model load.
        // Run in separate processes with optimization flags off/on (static policy).
        let data = Data("""
        {"hidden_size":64,"linear_num_value_heads":32,"linear_num_key_heads":16,
         "linear_key_head_dim":128,"linear_value_head_dim":128,
         "linear_conv_kernel_dim":4,"rms_norm_eps":0.000001}
        """.utf8)
        let config = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: data)
        // The original direct initializer remains source compatible.
        let layer = Qwen35GatedDeltaNet(config)
        layer.apply(map: { $0.asType(.bfloat16) })
        let input = (sin(MLXArray(0..<64).asType(.float32) * 0.173) * 0.03).asType(.bfloat16).reshaped(1, 1, 64)
        let fresh = MambaCache()
        let reference = layer(input, cache: fresh)
        MLX.eval(reference)
        XCTAssertEqual(reference.shape, [1, 1, 64])
        XCTAssertTrue(MLX.all(isFinite(reference)).item(Bool.self))
        for shape in [[1], [2, 32, 128, 128], [1, 31, 128, 128], [1, 32, 127, 128], [1, 32, 128, 127]] {
            let restored = MambaCache()
            restored[1] = MLXArray.zeros(shape, dtype: .float32)
            let output = layer(input, cache: restored)
            MLX.eval(output)
            XCTAssertTrue(MLX.all(output .== reference).item(Bool.self), "wrong restored shape: \(shape)")
            let state = try XCTUnwrap(restored[1])
            XCTAssertEqual(state.shape, [1, 32, 128, 128])
            XCTAssertEqual(state.dtype, .float32)
            XCTAssertEqual(restored.offset, fresh.offset)
        }
    }
}
