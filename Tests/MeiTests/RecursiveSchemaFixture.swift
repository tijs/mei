import Foundation
@testable import MeiCore

/// Shared fixture for the recursive strict-schema slice: one schema that
/// exercises nested objects, arrays with `items`, and a nullable union, plus
/// a toy token vocabulary that can spell its document and decoys for every
/// scoping and type boundary (nested keys, wrong item types, non-nullable
/// nulls, premature EOS).
enum RecursiveSchemaFixture {

    /// Nested object + array + nullable union under a strict root:
    /// `meta` is a nested strict object, `tags` an array of strings, and
    /// `note` a nullable string union.
    static let schemaJSON =
        #"{"type": "object", "properties": {"meta": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"], "additionalProperties": false}, "tags": {"type": "array", "items": {"type": "string"}}, "note": {"type": ["string", "null"]}}, "required": ["meta", "tags", "note"], "additionalProperties": false}"#

    /// The document the token script below spells.
    static let document = #"{"meta":{"id":7},"tags":["a","b"],"note":null}"#

    static func schema() throws -> CompiledJSONSchema {
        try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "nested_v1", strict: true,
                schema: try JSONDecoder().decode(MeiJSONValue.self, from: Data(schemaJSON.utf8))))
    }

    enum Token: Int, CaseIterable {
        case openMetaNested  // `{"meta":{`
        case idKey  // `"id":`
        case seven  // `7`
        case closeNested  // `}`
        case tagsArrayOpen  // `,"tags":[`
        case stringA  // `"a"`
        case stringB  // `"b"`
        case comma  // `,`
        case closeBracket  // `]`
        case noteKey  // `,"note":`
        case nullLiteral  // `null`
        case stringX  // `"x"`
        case closeRoot  // `}`
        case trueLiteral  // `true`
        case tagsKey  // `"tags":` — a root key, not a `meta` key
        case space  // ` `
        case wholeDocument  // the entire document in one token
        case eos  // no fragment

        var fragment: [UInt8]? {
            func bytes(_ string: String) -> [UInt8] { Array(string.utf8) }
            switch self {
            case .openMetaNested: return bytes("{\"meta\":{")
            case .idKey: return bytes("\"id\":")
            case .seven: return bytes("7")
            case .closeNested: return bytes("}")
            case .tagsArrayOpen: return bytes(",\"tags\":[")
            case .stringA: return bytes("\"a\"")
            case .stringB: return bytes("\"b\"")
            case .comma: return bytes(",")
            case .closeBracket: return bytes("]")
            case .noteKey: return bytes(",\"note\":")
            case .nullLiteral: return bytes("null")
            case .stringX: return bytes("\"x\"")
            case .closeRoot: return bytes("}")
            case .trueLiteral: return bytes("true")
            case .tagsKey: return bytes("\"tags\":")
            case .space: return bytes(" ")
            case .wholeDocument: return bytes(RecursiveSchemaFixture.document)
            case .eos: return nil
            }
        }
    }

    static func table() -> StaticTokenFragmentTable {
        StaticTokenFragmentTable(
            fragments: Token.allCases.map { $0.fragment },
            endOfSequenceTokenIds: [Token.eos.rawValue])
    }

    /// The tokens that spell `document` (decoys excluded), then EOS.
    static var documentScript: [Int] {
        [
            Token.openMetaNested, .idKey, .seven, .closeNested,
            .tagsArrayOpen, .stringA, .comma, .stringB, .closeBracket,
            .noteKey, .nullLiteral, .closeRoot, .eos,
        ].map(\.rawValue)
    }
}
