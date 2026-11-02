import Foundation
import XCTest
@testable import MeiCore

final class TextCompletionResponseTests: XCTestCase {
    func testRawTextEnvelopePreservesBytesAndUsage() throws {
        var run = GenerationRun()
        run.text = "raw \"text\"\n日本語"
        run.finishReason = "length"
        run.promptTokenCount = 12
        run.completionTokenCount = 8
        run.cachedTokenCount = 4
        run.decodeTokensPerSecond = 40
        let response = Router.textCompletionResponse(run: run, model: "fixture/raw")
        let data = try JSONEncoder().encode(response)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["object"] as? String, "text_completion")
        XCTAssertEqual(json["model"] as? String, "fixture/raw")
        XCTAssertTrue((json["id"] as? String)?.hasPrefix("cmpl-") == true)
        let choice = try XCTUnwrap((json["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice["text"] as? String, run.text)
        XCTAssertEqual(choice["index"] as? Int, 0)
        XCTAssertEqual(choice["finish_reason"] as? String, "length")
        XCTAssertNil(choice["message"])
        let usage = try XCTUnwrap(json["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 12)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 8)
        XCTAssertEqual(usage["total_tokens"] as? Int, 20)
        XCTAssertEqual(usage["tokens_per_second"] as? Double, 40)
        XCTAssertEqual((usage["prompt_tokens_details"] as? [String: Int])?["cached_tokens"], 4)
        let chat = Router.completionResponse(run: run, model: "fixture/raw", emitReasoning: true)
        XCTAssertEqual(chat.object, "chat.completion")
        XCTAssertEqual(chat.choices.first?.message.content, run.text)
    }
}
