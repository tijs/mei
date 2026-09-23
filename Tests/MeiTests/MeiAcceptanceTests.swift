import XCTest
import Foundation
import MeiCore

/// Black-box acceptance oracle for the Mei server (Phase 1, ported from
/// local-model-bench's runner/probe_omlx.py semantics).
///
/// These tests were written and run RED before the server existed. They hit a
/// live server at MEI_ACCEPTANCE_BASE_URL (default http://127.0.0.1:8024/v1).
/// Without a server, every probe fails with a connection error — that IS the
/// red state; the suite goes green only once the server behaves correctly.
final class MeiAcceptanceTests: XCTestCase {
    var baseURL: String {
        ProcessInfo.processInfo.environment["MEI_ACCEPTANCE_BASE_URL"] ?? "http://127.0.0.1:8024/v1"
    }

    var modelID: String {
        ProcessInfo.processInfo.environment["MEI_SERVED_MODEL_ID"] ?? "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"
    }

    var timeout: TimeInterval {
        TimeInterval(ProcessInfo.processInfo.environment["MEI_ACCEPTANCE_TIMEOUT"] ?? "1800") ?? 1800
    }

    // MARK: - Helpers

    /// Concatenate the base URL and a route, tolerating a leading slash on
    /// the route (a naive "\(baseURL)/\(route)" would double the slash and
    /// 404 against the router's exact-match URI table).
    func resolve(_ route: String) -> URL {
        let trimmed = route.hasPrefix("/") ? String(route.dropFirst()) : route
        return URL(string: "\(baseURL)/\(trimmed)")!
    }

    func post(_ route: String, json: [String: Any], timeout: TimeInterval? = nil) throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: resolve(route))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: json)
        return try send(request, timeout: timeout)
    }

    func get(_ route: String, timeout: TimeInterval? = nil) throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: resolve(route))
        request.httpMethod = "GET"
        return try send(request, timeout: timeout)
    }

    func send(_ request: URLRequest, timeout: TimeInterval?) throws -> (Data, HTTPURLResponse) {
        let (data, response) = try URLSession.shared.synchronousData(for: request, timeout: timeout ?? self.timeout)
        guard let http = response as? HTTPURLResponse else {
            throw XCTSkip("non-HTTP response")
        }
        return (data, http)
    }

    func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        try JSONDecoder().decode(T.self, from: data)
    }

    func dict(_ data: Data) throws -> [String: Any] {
        (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    // MARK: - Probes

    func testModelsIdentity() throws {
        let (data, response) = try get("/models", timeout: 30)
        XCTAssertEqual(response.statusCode, 200, String(data: data, encoding: .utf8) ?? "")
        let body = try dict(data)
        let ids = (body["data"] as? [[String: Any]])?.compactMap { $0["id"] as? String } ?? []
        XCTAssertTrue(ids.contains(modelID), "exact model ID \(modelID) absent from \(ids)")
    }

    func testPlainCompletionNonStreaming() throws {
        let (data, response) = try post("/chat/completions", json: [
            "model": modelID,
            "messages": [["role": "user", "content": "Reply with exactly: ready"]],
            "temperature": 0,
            // Thinking models (Ornith is Qwen3.5-lineage) spend their first
            // 100+ tokens on the thinking preamble; a small budget truncates
            // before any visible content. 1024 covers a full think+answer.
            "max_tokens": 1024,
            "stream": false,
        ])
        XCTAssertEqual(response.statusCode, 200, String(data: data, encoding: .utf8) ?? "")
        let body = try dict(data)
        let content = (((body["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String) ?? ""
        XCTAssertFalse(content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "plain completion returned empty content")
    }

    func testToolCallNonStreaming() throws {
        let toolPayload: [String: Any] = [
            "model": modelID,
            "messages": [
                ["role": "system", "content": "You are a helpful assistant with access to one tool."],
                ["role": "user", "content": "What is 15 + 27? You must use add_numbers to compute it."],
            ],
            "tools": [[
                "type": "function",
                "function": [
                    "name": "add_numbers",
                    "description": "Adds two numbers and returns the sum.",
                    "parameters": [
                        "type": "object",
                        "properties": ["a": ["type": "number"], "b": ["type": "number"]],
                        "required": ["a", "b"],
                        "additionalProperties": false,
                    ],
                ],
            ]],
            "tool_choice": ["type": "function", "function": ["name": "add_numbers"]],
            "temperature": 0,
            "max_tokens": 256,
            "stream": false,
        ]
        let (data, response) = try post("/chat/completions", json: toolPayload)
        XCTAssertEqual(response.statusCode, 200, String(data: data, encoding: .utf8) ?? "")
        let body = try dict(data)
        try validateAddCall(body)
    }

    func testToolCallStreaming() throws {
        var toolPayload: [String: Any] = [
            "model": modelID,
            "messages": [
                ["role": "system", "content": "You are a helpful assistant with access to one tool."],
                ["role": "user", "content": "What is 15 + 27? You must use add_numbers to compute it."],
            ],
            "tools": [[
                "type": "function",
                "function": [
                    "name": "add_numbers",
                    "description": "Adds two numbers and returns the sum.",
                    "parameters": [
                        "type": "object",
                        "properties": ["a": ["type": "number"], "b": ["type": "number"]],
                        "required": ["a", "b"],
                        "additionalProperties": false,
                    ],
                ],
            ]],
            "tool_choice": ["type": "function", "function": ["name": "add_numbers"]],
            "temperature": 0,
            "max_tokens": 256,
            "stream": true,
        ]
        toolPayload["stream_options"] = ["include_usage": true]
        let url = URL(string: "\(baseURL)/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: toolPayload)
        let (data, response) = try send(request, timeout: nil)
        XCTAssertEqual(response.statusCode, 200, String(data: data, encoding: .utf8) ?? "")
        let assembled = try assembleSSE(data)
        try validateAddCall(assembled)
    }

    func validateAddCall(_ body: [String: Any]) throws {
        let choices = body["choices"] as? [[String: Any]] ?? []
        let message = choices.first?["message"] as? [String: Any] ?? [:]
        let calls = message["tool_calls"] as? [[String: Any]] ?? []
        XCTAssertEqual(calls.count, 1, "expected one structured tool call, got \(calls)")
        let function = calls.first?["function"] as? [String: Any] ?? [:]
        XCTAssertEqual(function["name"] as? String, "add_numbers")
        let arguments = try XCTUnwrap(function["arguments"] as? String)
        let parsed = (try? JSONSerialization.jsonObject(with: Data(arguments.utf8)) as? [String: Any]) ?? [:]
        func num(_ value: Any?) -> Double {
            (value as? NSNumber)?.doubleValue ?? -1
        }
        XCTAssertEqual(num(parsed["a"]), 15)
        XCTAssertEqual(num(parsed["b"]), 27)
        XCTAssertEqual(choices.first?["finish_reason"] as? String, "tool_calls")
    }

    /// Reassemble an SSE body into a single completion-shaped dictionary,
    /// delegating to the deterministic `SSEFrameParser` (the same merge the
    /// unit tests pin: index-keyed tool-call fragments, role/content
    /// accumulation, usage capture).
    func assembleSSE(_ data: Data) throws -> [String: Any] {
        let result = SSEFrameParser.parse(data)
        var message: [String: Any] = ["role": "assistant"]
        if !result.content.isEmpty { message["content"] = result.content }
        if !result.toolCalls.isEmpty {
            message["tool_calls"] = result.toolCalls.map { call in
                var part: [String: Any] = [
                    "type": "function",
                    "function": ["name": call.name, "arguments": call.arguments],
                ]
                if let id = call.id { part["id"] = id }
                return part
            }
        }
        var body: [String: Any] = [
            "choices": [["message": message, "finish_reason": result.finishReason ?? "stop"]],
        ]
        if let usage = result.usage {
            body["usage"] = [
                "prompt_tokens": usage.promptTokens,
                "completion_tokens": usage.completionTokens,
                "total_tokens": usage.totalTokens,
                "prompt_tokens_details": ["cached_tokens": usage.cachedTokens ?? 0],
            ]
        }
        return body
    }

    func testHealthEndpoint() throws {
        // /healthz is a root-level route (not under /v1); derive the server
        // root from the configured base URL.
        let base = baseURL.hasSuffix("/v1") ? String(baseURL.dropLast(3)) : baseURL
        let url = URL(string: "\(base)/healthz")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        let (data, response) = try send(request, timeout: 30)
        XCTAssertEqual(response.statusCode, 200, String(data: data, encoding: .utf8) ?? "")
        let body = try dict(data)
        XCTAssertEqual(body["status"] as? String, "ok")
    }
}

extension URLSession {
    func synchronousData(for request: URLRequest, timeout: TimeInterval) throws -> (Data, URLResponse) {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<(Data, URLResponse), Error>?
        let task = self.dataTask(with: request) { data, response, error in
            if let error {
                result = .failure(error)
            } else if let data, let response {
                result = .success((data, response))
            } else {
                result = .failure(URLError(.badServerResponse))
            }
            semaphore.signal()
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        task.cancel()
        return try result!.get()
    }
}