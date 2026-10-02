import Foundation
import XCTest

/// Exact CoCore attached-engine canary fixtures — the acceptance oracle for
/// `MeiAcceptanceTests` and the model-free pipeline tests.
///
/// Mirrors the Rust source of graze-social/cocore PR #237 (merged as
/// `0151475bf8c98de10a64cab51c23a46dd84a8fe1`,
/// `provider/src/engines/openai_http.rs`):
///
/// - `structuredOutputBody(model:)` mirrors `structured_output_canary_body`
///   field-for-field (messages, `response_format` envelope, `max_tokens` 64,
///   `temperature` 0; no `stream` key — the canary POSTs non-streaming).
/// - `structuredOutputPassed(content:)` mirrors
///   `structured_output_canary_passed`: the whole `choices[0].message.content`
///   parses as a JSON object whose `status` is exactly `"ok"` and which has no
///   other keys.
///
/// Keep in sync with docs/COCORE.md §2; do not "improve" the shapes here —
/// they are what the live server is judged against.
enum CoCoreCanary {

    /// The exact expected passing content.
    static let passingContent = #"{"status":"ok"}"#

    /// The exact non-streaming request body CoCore POSTs to
    /// `/v1/chat/completions` at startup. The prompt deliberately begs for
    /// prose so a server that silently drops `response_format` fails the
    /// canary instead of passing by luck.
    static func structuredOutputBody(model: String) -> [String: Any] {
        [
            "model": model,
            "messages": [
                [
                    "role": "system",
                    "content": "You are a friendly assistant who always answers in two or three warm, conversational sentences.",
                ],
                [
                    "role": "user",
                    "content": "Say hello and tell me how you are doing today.",
                ],
            ],
            "response_format": [
                "type": "json_schema",
                "json_schema": [
                    "name": "canary_status",
                    "strict": true,
                    "schema": [
                        "type": "object",
                        "properties": [
                            "status": [
                                "type": "string",
                                "enum": ["ok"],
                            ] as [String: Any]
                        ],
                        "required": ["status"],
                        "additionalProperties": false,
                    ] as [String: Any],
                ] as [String: Any],
            ] as [String: Any],
            "max_tokens": 64,
            "temperature": 0,
        ]
    }

    /// The request body as wire bytes (sorted keys, like CoCore's serde_json
    /// default map ordering).
    static func structuredOutputBodyData(model: String) -> Data {
        (try? JSONSerialization.data(
            withJSONObject: structuredOutputBody(model: model), options: [.sortedKeys]))
            ?? Data()
    }

    /// The CoCore pass condition applied to a decoded `message.content`.
    static func structuredOutputPassed(content: String?) -> Bool {
        guard let content else { return false }
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8),
            let value = try? JSONSerialization.jsonObject(with: data),
            let object = value as? [String: Any]
        else {
            return false
        }
        return object.count == 1 && object["status"] as? String == "ok"
    }

    /// The CoCore pass condition applied to a decoded non-streaming response
    /// body (`choices[0].message.content`).
    static func structuredOutputPassed(responseBody: [String: Any]) -> Bool {
        guard let choices = responseBody["choices"] as? [[String: Any]],
            let message = choices.first?["message"] as? [String: Any]
        else {
            return false
        }
        return structuredOutputPassed(content: message["content"] as? String)
    }

    /// A minimal response body shaped like Mei's non-streaming response, for
    /// checker tests (mirrors cocore's `so_response` test helper).
    static func responseBody(content: String?, finishReason: String = "stop") -> [String: Any] {
        var message: [String: Any] = ["role": "assistant"]
        if let content { message["content"] = content }
        return ["choices": [["message": message, "finish_reason": finishReason]]]
    }
}
