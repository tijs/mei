import Foundation

/// Maps OpenAI chat-API messages onto chat-template dictionaries. The shape
/// mirrors what vmlx-swift's own DefaultMessageGenerator emits: tool_calls
/// entries carry BOTH a top-level `name`/`arguments` view AND a nested
/// `function.name`/`function.arguments` view, because real templates read
/// either convention.
public enum MessageMapping {
    public static func templateDictionary(
        from message: APIMessage
    ) -> [String: any Sendable] {
        var dict: [String: any Sendable] = ["role": message.role]
        if let content = message.content, !content.isEmpty {
            dict["content"] = content
        }
        if let reasoning = message.reasoningContent, !reasoning.isEmpty {
            dict["reasoning_content"] = reasoning
        }
        switch message.role {
        case "tool":
            dict["tool_call_id"] = message.toolCallID ?? ""
        case "assistant":
            if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
                dict["tool_calls"] = toolCalls.map { call -> [String: any Sendable] in
                    let arguments = MeiJSONValue.parseObject(from: call.argumentsJSON) ?? .object([:])
                    let argumentsSendable = MeiJSONValue.templateSendable(arguments)
                    let functionView: [String: any Sendable] = [
                        "name": call.name,
                        "arguments": argumentsSendable,
                    ]
                    var entry: [String: any Sendable] = [
                        "name": call.name,
                        "arguments": argumentsSendable,
                        "function": functionView,
                    ]
                    if let id = call.id {
                        entry["id"] = id
                        entry["type"] = "function"
                    }
                    return entry
                }
            }
        default:
            break
        }
        return dict
    }

    public static func templateTools(_ tools: [MeiJSONValue]?) -> [[String: any Sendable]]? {
        guard let tools, !tools.isEmpty else { return nil }
        return tools.map { tool in
            guard case .object(let object) = tool else {
                return ["type": "function"]
            }
            var dict: [String: any Sendable] = [:]
            for (key, value) in object {
                dict[key] = MeiJSONValue.templateSendable(value)
            }
            return dict
        }
    }

    /// additionalContext passed to the chat template for reasoning toggles
    /// and forced tool selection.
    public static func additionalContext(
        enableThinking: Bool?,
        reasoningEffort: String?,
        toolChoice: MeiJSONValue?
    ) -> [String: any Sendable]? {
        var context: [String: any Sendable] = [:]
        if let enableThinking {
            context["enable_thinking"] = enableThinking
        }
        if let reasoningEffort {
            context["reasoning_effort"] = reasoningEffort
        }
        if let toolChoice {
            switch toolChoice {
            case .string(let name):
                if name == "required" || name == "auto" || name == "none" {
                    context["tool_choice"] = name
                } else {
                    context["tool_choice"] = "required"
                    context["tool_choice_name"] = name
                }
            case .object(let object):
                if case .string(let name)? = object["name"] {
                    context["tool_choice"] = "required"
                    context["tool_choice_name"] = name
                } else {
                    context["tool_choice"] = "required"
                }
            default:
                break
            }
        }
        return context.isEmpty ? nil : context
    }
}