import Foundation
@testable import MeiCore

/// Shared fixture for the schema-matrix slice: one strict schema exercising
/// the expanded constraint surface — integer bounds + `multipleOf`, number
/// exclusive bounds + fractional `multipleOf`, string/integer/boolean enums,
/// a nullable integer enum, and array `minItems`/`maxItems` — plus a toy
/// token vocabulary that can spell its document and decoys for every
/// constraint boundary (out-of-range, non-multiple, wrong enum member,
/// below `minItems`, above `maxItems`).
enum SchemaMatrixFixture {

    /// `level`: integer in [10, 99], multiple of 5.
    /// `ratio`: number in (0, 1), multiple of 0.25.
    /// `mode`: string enum ["fast", "slow"].
    /// `retries`: nullable integer enum [0, 1, 2, null].
    /// `ok`: boolean enum [true].
    /// `flags`: 1–2 booleans.
    static let schemaJSON =
        #"{"type": "object", "properties": {"level": {"type": "integer", "minimum": 10, "maximum": 99, "multipleOf": 5}, "ratio": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1, "multipleOf": 0.25}, "mode": {"type": "string", "enum": ["fast", "slow"]}, "retries": {"type": ["integer", "null"], "enum": [0, 1, 2, null]}, "ok": {"type": "boolean", "enum": [true]}, "flags": {"type": "array", "items": {"type": "boolean"}, "minItems": 1, "maxItems": 2}}, "required": ["level", "ratio", "mode", "retries", "ok", "flags"], "additionalProperties": false}"#

    /// The document the token script below spells.
    static let document =
        #"{"level":15,"ratio":0.5,"mode":"fast","retries":null,"ok":true,"flags":[true,false]}"#

    static func schema() throws -> CompiledJSONSchema {
        try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "matrix_v1", strict: true,
                schema: try JSONDecoder().decode(MeiJSONValue.self, from: Data(schemaJSON.utf8))))
    }

    enum Token: Int, CaseIterable {
        case openLevel  // `{"level":`
        case fifteen  // `15` — in range, multiple of 5
        case twenty  // `20` — in range, multiple of 5
        case twelve  // `12` — in range but not a multiple of 5
        case ratioKey  // `,"ratio":`
        case zeroPointFive  // `0.5` — in (0, 1), multiple of 0.25
        case zeroPointThree  // `0.3` — in (0, 1) but not a multiple of 0.25
        case zero  // `0` — at the exclusive lower bound
        case one  // `1` — at the exclusive upper bound
        case modeKey  // `,"mode":`
        case fastString  // `"fast"` — enum member
        case slowString  // `"slow"` — enum member
        case nopeString  // `"nope"` — not an enum member
        case retriesKey  // `,"retries":`
        case two  // `2` — enum member
        case three  // `3` — not an enum member
        case nullLiteral  // `null`
        case okKey  // `,"ok":`
        case trueLiteral  // `true`
        case falseLiteral  // `false`
        case flagsOpen  // `,"flags":[`
        case comma  // `,`
        case closeBracket  // `]`
        case closeRoot  // `}`
        case space  // ` `
        case wholeDocument  // the entire document in one token
        case eos  // no fragment

        var fragment: [UInt8]? {
            func bytes(_ string: String) -> [UInt8] { Array(string.utf8) }
            switch self {
            case .openLevel: return bytes("{\"level\":")
            case .fifteen: return bytes("15")
            case .twenty: return bytes("20")
            case .twelve: return bytes("12")
            case .ratioKey: return bytes(",\"ratio\":")
            case .zeroPointFive: return bytes("0.5")
            case .zeroPointThree: return bytes("0.3")
            case .zero: return bytes("0")
            case .one: return bytes("1")
            case .modeKey: return bytes(",\"mode\":")
            case .fastString: return bytes("\"fast\"")
            case .slowString: return bytes("\"slow\"")
            case .nopeString: return bytes("\"nope\"")
            case .retriesKey: return bytes(",\"retries\":")
            case .two: return bytes("2")
            case .three: return bytes("3")
            case .nullLiteral: return bytes("null")
            case .okKey: return bytes(",\"ok\":")
            case .trueLiteral: return bytes("true")
            case .falseLiteral: return bytes("false")
            case .flagsOpen: return bytes(",\"flags\":[")
            case .comma: return bytes(",")
            case .closeBracket: return bytes("]")
            case .closeRoot: return bytes("}")
            case .space: return bytes(" ")
            case .wholeDocument: return bytes(SchemaMatrixFixture.document)
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
            Token.openLevel, .fifteen, .ratioKey, .zeroPointFive,
            .modeKey, .fastString, .retriesKey, .nullLiteral,
            .okKey, .trueLiteral, .flagsOpen, .trueLiteral,
            .comma, .falseLiteral, .closeBracket, .closeRoot,
        ].map(\.rawValue) + [Token.eos.rawValue]
    }
}
