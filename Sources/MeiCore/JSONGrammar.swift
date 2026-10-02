import Foundation

// MARK: - Model-free JSON grammar automaton

/// Deterministic byte-level automaton for the JSON language constrained
/// generation must produce.
///
/// This is the model- and tokenizer-free half of the constrained-decoding
/// core: callers feed it the bytes of sampled token fragments (or single
/// bytes) and it tracks the grammar state, reporting `.complete` once a full
/// root JSON value has been consumed and `.failed` after any illegal byte.
/// The token-level mask (`JSONGrammarProcessor`) drives it with a caller-
/// supplied vocabulary; production tokenizer/model wiring stays outside this
/// boundary so the automaton can be fuzzed and reused by buffered and streamed
/// generation.
///
/// Supported shapes:
/// - `json_object`: any complete JSON value (object, array, string with
///   escapes, number, boolean, null), with a nesting cap;
/// - strict `json_schema`: the flat object subset compiled by
///   `JSONSchemaCompiler` — scalar properties with optional string enums,
///   `required` keys enforced at close, `additionalProperties: false`, keys in
///   any order (each at most once).
///
/// Guarantees:
/// - every accepted root value is syntactically valid JSON: strict UTF-8
///   inside strings, `\uXXXX` escapes with enforced surrogate pairing, no raw
///   control characters, JSON number syntax, and legal JSON whitespace only;
/// - acceptance is reported only after the root value is complete, so
///   end-of-sequence can never be sampled mid-document;
/// - an illegal byte fails the state closed (sticky until `reset()`), which
///   the token-level mask turns into an explicit generation error rather than
///   a "successful" response containing invalid JSON.
public struct JSONGrammarState: Sendable, Equatable {

    /// Default cap on nested objects/arrays in `json_object` mode. Deeper
    /// input fails closed (the opening bracket is rejected) instead of being
    /// unbounded; schemas with nested values are not part of the supported
    /// subset yet.
    public static let defaultMaximumNestingDepth = 32

    // MARK: Configuration

    private enum GrammarKind: Equatable {
        case freeJSON
        case strictSchema(CompiledJSONSchema)
    }

    private let kind: GrammarKind
    private let maximumNestingDepth: Int

    private var schema: CompiledJSONSchema? {
        if case .strictSchema(let schema) = kind { return schema }
        return nil
    }

    // MARK: Mutable state

    private var phase: Phase
    private var stack: [Frame]

    // MARK: Init

    /// Builds the automaton for a compiled response format.
    ///
    /// `.text` has no grammar (it must not constrain generation), so it throws
    /// instead of silently producing an inert constraint.
    public init(
        format: CompiledResponseFormat,
        maximumNestingDepth: Int = JSONGrammarState.defaultMaximumNestingDepth
    ) throws {
        let kind: GrammarKind
        switch format {
        case .text:
            throw JSONGrammarError.unconstrainedFormat
        case .jsonObject:
            kind = .freeJSON
        case .jsonSchema(let compiled):
            kind = .strictSchema(compiled)
        }
        self.init(kind: kind, maximumNestingDepth: maximumNestingDepth)
    }

    /// Free JSON value mode (`response_format: {"type": "json_object"}`).
    public init(jsonObjectWithMaximumNestingDepth maximumNestingDepth: Int = JSONGrammarState.defaultMaximumNestingDepth) {
        self.init(kind: .freeJSON, maximumNestingDepth: maximumNestingDepth)
    }

    /// Strict flat-object schema mode.
    public init(
        schema: CompiledJSONSchema,
        maximumNestingDepth: Int = JSONGrammarState.defaultMaximumNestingDepth
    ) {
        self.init(kind: .strictSchema(schema), maximumNestingDepth: maximumNestingDepth)
    }

    private init(kind: GrammarKind, maximumNestingDepth: Int) {
        self.kind = kind
        self.maximumNestingDepth = Swift.max(1, maximumNestingDepth)
        self.stack = []
        self.phase = Self.initialPhase(for: kind)
    }

    // MARK: Status

    /// Grammar status at the current position.
    public var status: JSONGrammarStatus {
        switch phase {
        case .failed:
            return .failed
        case .finished:
            return .finished
        case .complete:
            return .complete
        case .number(let numberState, _) where stack.isEmpty && numberState.isTerminable:
            return .complete
        case .literal(let literal) where stack.isEmpty && literal.isComplete:
            return .complete
        default:
            return .inProgress
        }
    }

    /// True when the root value is complete: end-of-sequence may be sampled.
    public var canAcceptEndOfSequence: Bool { status == .complete }

    /// Consumes the end-of-sequence signal. Returns false (and leaves the
    /// state untouched) unless the root value is complete.
    @discardableResult
    public mutating func markEndOfSequence() -> Bool {
        guard status == .complete else { return false }
        phase = .finished
        return true
    }

    /// Resets to the initial state — the prompt-reset hook for a new request.
    public mutating func reset() {
        stack.removeAll()
        phase = Self.initialPhase(for: kind)
    }

    // MARK: Byte consumption

    /// Consumes one byte. Returns false when the byte is illegal in the
    /// current state, in which case the state is failed closed until `reset()`.
    @discardableResult
    public mutating func consume(byte: UInt8) -> Bool {
        guard phase != .failed, phase != .finished else { return false }
        var byte = byte
        while true {
            switch step(byte) {
            case .consumed:
                return true
            case .retry:
                continue
            case .rejected:
                phase = .failed
                return false
            }
        }
    }

    /// Consumes a token fragment. All-or-nothing from the caller's point of
    /// view: the first rejected byte fails the state closed. Trial checks use
    /// a value copy instead of this method.
    @discardableResult
    public mutating func consume(fragment: [UInt8]) -> Bool {
        for byte in fragment where !consume(byte: byte) { return false }
        return true
    }

    // MARK: - Phase machine

    private enum StepOutcome {
        case consumed
        /// The byte was not consumed in the previous phase; reprocess it in
        /// the new phase (used when a number/literal ends at a terminator).
        case retry
        case rejected
    }

    private enum ValueConstraint: Equatable {
        /// Any JSON value (`json_object` mode).
        case anyJSON
        /// The root object of a strict schema.
        case schemaRootObject
        /// A scalar property of a strict schema.
        case schemaScalar(CompiledJSONSchema.Property)
    }

    private enum StringRole: Equatable {
        /// Unconstrained string content of a string value.
        case plain
        /// A free-object key: any string, followed by `:`.
        case freeKey
        /// A schema object key: must complete to an unused defined property.
        case key(ScalarPrefixMatcher)
        /// A string enum value: must complete to one of the allowed values.
        case enumValue(ScalarPrefixMatcher)
    }

    private struct ScalarPrefixMatcher: Equatable {
        private struct Candidate: Equatable {
            let name: String
            let scalars: [Unicode.Scalar]
        }

        private let candidates: [Candidate]
        private var decoded: [Unicode.Scalar] = []

        init(_ names: [String]) {
            candidates = names.map { Candidate(name: $0, scalars: Array($0.unicodeScalars)) }
        }

        /// Appends one decoded scalar; false when no candidate keeps the prefix.
        mutating func append(_ scalar: Unicode.Scalar) -> Bool {
            decoded.append(scalar)
            return candidates.contains { candidate in
                candidate.scalars.count >= decoded.count && Array(candidate.scalars.prefix(decoded.count)) == decoded
            }
        }

        var isComplete: Bool { candidates.contains { $0.scalars == decoded } }

        var completedName: String? { candidates.first { $0.scalars == decoded }?.name }

        /// True when some candidate's next scalar could fall inside `range`.
        /// Checked before committing to a multi-byte UTF-8 sequence or a
        /// surrogate pair so no fragment can lead into a dead-end state.
        func canAppend(scalarIn range: ClosedRange<UInt32>) -> Bool {
            candidates.contains { candidate in
                candidate.scalars.count > decoded.count && range.contains(candidate.scalars[decoded.count].value)
            }
        }
    }

    private struct UTF8Sequence: Equatable {
        /// Scalar accumulated so far (continuation bits are OR'd in).
        var scalar: UInt32
        /// Continuation bytes still expected.
        var remaining: Int
        /// Valid range for the next continuation byte.
        var nextByteRange: ClosedRange<UInt8>
        /// Range of scalars this sequence can decode to.
        var scalarRange: ClosedRange<UInt32>
    }

    private enum NumberState: Equatable {
        case sign
        case zero
        case integerDigits
        case fractionStart
        case fractionDigits
        case exponentStart
        case exponentSign
        case exponentDigits

        var isTerminable: Bool {
            switch self {
            case .zero, .integerDigits, .fractionDigits, .exponentDigits: return true
            case .sign, .fractionStart, .exponentStart, .exponentSign: return false
            }
        }
    }

    private enum NumberKind: Equatable {
        case number
        case integer
    }

    private struct LiteralState: Equatable {
        let expected: [UInt8]
        var matched: Int

        var isComplete: Bool { matched == expected.count }
    }

    private enum Frame: Equatable {
        case freeObject
        case freeArray
        case schemaObject(used: Set<String>)
    }

    private enum Phase: Equatable {
        /// Expecting the start of a value. `allowsEmptyClose` is true only
        /// directly after `[` (an empty array may close immediately).
        case valueStart(ValueConstraint, allowsEmptyClose: Bool)
        /// Expecting an object key. `fresh` is true only directly after `{`.
        case keyStart(fresh: Bool)
        /// An object key completed; expecting `:`. Carries the constraint for
        /// the member value (`.anyJSON` for free objects, the scalar property
        /// for schema objects).
        case expectColon(ValueConstraint)
        /// A value completed inside a container; expecting `,` or the closer.
        case afterValue
        /// Inside a string.
        case string(StringRole, utf8: UTF8Sequence?)
        /// Saw `\`; expecting an escape character.
        case stringEscape(StringRole)
        /// Inside a `\uXXXX` escape.
        case unicodeEscape(StringRole, nibbles: Int, value: UInt32)
        /// A high-surrogate escape completed; expecting `\u` plus a low
        /// surrogate. `offset` 0 expects `\`, 1 expects `u`, 2...5 are the
        /// constrained hex nibbles.
        case lowSurrogateEscape(StringRole, high: UInt32, offset: Int, value: UInt32)
        /// Inside a number.
        case number(NumberState, NumberKind)
        /// Inside a `true`/`false`/`null` literal.
        case literal(LiteralState)
        /// Root value complete; only JSON whitespace may follow.
        case complete
        /// End-of-sequence consumed.
        case finished
        /// An illegal byte was rejected; sticky until `reset()`.
        case failed
    }

    private static func initialPhase(for kind: GrammarKind) -> Phase {
        switch kind {
        case .freeJSON:
            return .valueStart(.anyJSON, allowsEmptyClose: false)
        case .strictSchema:
            return .valueStart(.schemaRootObject, allowsEmptyClose: false)
        }
    }

    private mutating func step(_ byte: UInt8) -> StepOutcome {
        switch phase {
        case .failed, .finished:
            return .rejected

        case .complete:
            return Self.isJSONWhitespace(byte) ? .consumed : .rejected

        case .valueStart(let constraint, let allowsEmptyClose):
            if Self.isJSONWhitespace(byte) { return .consumed }
            if allowsEmptyClose, byte == Self.closeBracket {
                return closeContainer(byte)
            }
            switch constraint {
            case .anyJSON:
                return startJSONValue(byte)
            case .schemaRootObject:
                guard byte == Self.openBrace else { return .rejected }
                stack.append(.schemaObject(used: []))
                phase = .keyStart(fresh: true)
                return .consumed
            case .schemaScalar(let property):
                return startScalarValue(byte, property: property)
            }

        case .keyStart(let fresh):
            if Self.isJSONWhitespace(byte) { return .consumed }
            if byte == Self.quote {
                startStringKey()
                return .consumed
            }
            if fresh, byte == Self.closeBrace {
                return closeContainer(byte)
            }
            return .rejected

        case .expectColon(let constraint):
            if Self.isJSONWhitespace(byte) { return .consumed }
            guard byte == Self.colon else { return .rejected }
            phase = .valueStart(constraint, allowsEmptyClose: false)
            return .consumed

        case .afterValue:
            if Self.isJSONWhitespace(byte) { return .consumed }
            if byte == Self.comma {
                guard canAddMember else { return .rejected }
                openNextMember()
                return .consumed
            }
            if byte == Self.closeBrace || byte == Self.closeBracket {
                return closeContainer(byte)
            }
            return .rejected

        case .string(let role, let utf8):
            return consumeStringByte(byte, role: role, utf8: utf8)

        case .stringEscape(let role):
            return consumeEscapeByte(byte, role: role)

        case .unicodeEscape(let role, let nibbles, let value):
            return consumeUnicodeEscapeByte(byte, role: role, nibbles: nibbles, value: value)

        case .lowSurrogateEscape(let role, let high, let offset, let value):
            return consumeLowSurrogateByte(byte, role: role, high: high, offset: offset, value: value)

        case .number(let numberState, let kind):
            return consumeNumberByte(byte, state: numberState, kind: kind)

        case .literal(let literal):
            return consumeLiteralByte(byte, literal: literal)
        }
    }

    // MARK: Value starts

    private mutating func startJSONValue(_ byte: UInt8) -> StepOutcome {
        if byte == Self.openBrace {
            guard stack.count < maximumNestingDepth else { return .rejected }
            stack.append(.freeObject)
            phase = .keyStart(fresh: true)
            return .consumed
        }
        if byte == Self.openBracket {
            guard stack.count < maximumNestingDepth else { return .rejected }
            stack.append(.freeArray)
            phase = .valueStart(.anyJSON, allowsEmptyClose: true)
            return .consumed
        }
        if byte == Self.quote {
            phase = .string(.plain, utf8: nil)
            return .consumed
        }
        if byte == Self.minus {
            phase = .number(.sign, .number)
            return .consumed
        }
        if byte == Self.zero {
            phase = .number(.zero, .number)
            return .consumed
        }
        if Self.isNonZeroDigit(byte) {
            phase = .number(.integerDigits, .number)
            return .consumed
        }
        if byte == UInt8(ascii: "t") {
            phase = .literal(LiteralState(expected: Self.trueLiteral, matched: 1))
            return .consumed
        }
        if byte == UInt8(ascii: "f") {
            phase = .literal(LiteralState(expected: Self.falseLiteral, matched: 1))
            return .consumed
        }
        if byte == UInt8(ascii: "n") {
            phase = .literal(LiteralState(expected: Self.nullLiteral, matched: 1))
            return .consumed
        }
        return .rejected
    }

    private mutating func startScalarValue(_ byte: UInt8, property: CompiledJSONSchema.Property) -> StepOutcome {
        switch property.type {
        case .string:
            guard byte == Self.quote else { return .rejected }
            if let allowedValues = property.allowedValues {
                phase = .string(.enumValue(ScalarPrefixMatcher(allowedValues)), utf8: nil)
            } else {
                phase = .string(.plain, utf8: nil)
            }
            return .consumed
        case .number:
            return startNumber(byte, kind: .number)
        case .integer:
            return startNumber(byte, kind: .integer)
        case .boolean:
            if byte == UInt8(ascii: "t") {
                phase = .literal(LiteralState(expected: Self.trueLiteral, matched: 1))
                return .consumed
            }
            if byte == UInt8(ascii: "f") {
                phase = .literal(LiteralState(expected: Self.falseLiteral, matched: 1))
                return .consumed
            }
            return .rejected
        }
    }

    private mutating func startNumber(_ byte: UInt8, kind: NumberKind) -> StepOutcome {
        if byte == Self.minus {
            phase = .number(.sign, kind)
            return .consumed
        }
        if byte == Self.zero {
            phase = .number(.zero, kind)
            return .consumed
        }
        if Self.isNonZeroDigit(byte) {
            phase = .number(.integerDigits, kind)
            return .consumed
        }
        return .rejected
    }

    // MARK: Keys and containers

    private mutating func startStringKey() {
        guard let frame = stack.last else {
            phase = .failed
            return
        }
        switch frame {
        case .freeObject:
            phase = .string(.freeKey, utf8: nil)
        case .schemaObject(let used):
            let unused = (schema?.properties.map(\.name) ?? []).filter { !used.contains($0) }
            phase = .string(.key(ScalarPrefixMatcher(unused)), utf8: nil)
        case .freeArray:
            phase = .failed
        }
    }

    private mutating func markKeyUsed(_ name: String) -> CompiledJSONSchema.Property? {
        guard let schema, let property = schema.properties.first(where: { $0.name == name }) else { return nil }
        guard case .schemaObject(var used)? = stack.last else { return nil }
        used.insert(name)
        stack[stack.count - 1] = .schemaObject(used: used)
        return property
    }

    private var canAddMember: Bool {
        guard let frame = stack.last else { return false }
        switch frame {
        case .freeObject, .freeArray:
            return true
        case .schemaObject(let used):
            guard let schema else { return false }
            return schema.properties.contains { !used.contains($0.name) }
        }
    }

    private mutating func openNextMember() {
        guard let frame = stack.last else {
            phase = .failed
            return
        }
        switch frame {
        case .freeObject, .schemaObject:
            phase = .keyStart(fresh: false)
        case .freeArray:
            phase = .valueStart(.anyJSON, allowsEmptyClose: false)
        }
    }

    private mutating func closeContainer(_ byte: UInt8) -> StepOutcome {
        guard let frame = stack.last else { return .rejected }
        switch frame {
        case .freeObject:
            guard byte == Self.closeBrace else { return .rejected }
            stack.removeLast()
            phase = valueCompletedPhase()
            return .consumed
        case .freeArray:
            guard byte == Self.closeBracket else { return .rejected }
            stack.removeLast()
            phase = valueCompletedPhase()
            return .consumed
        case .schemaObject(let used):
            guard byte == Self.closeBrace else { return .rejected }
            guard let schema, schema.required.allSatisfy(used.contains) else { return .rejected }
            stack.removeLast()
            phase = valueCompletedPhase()
            return .consumed
        }
    }

    private func valueCompletedPhase() -> Phase {
        stack.isEmpty ? .complete : .afterValue
    }

    // MARK: Strings

    private mutating func consumeStringByte(_ byte: UInt8, role: StringRole, utf8: UTF8Sequence?) -> StepOutcome {
        if let pending = utf8 {
            guard pending.nextByteRange.contains(byte) else { return .rejected }
            let value = (pending.scalar << 6) | UInt32(byte & 0x3F)
            if pending.remaining > 1 {
                phase = .string(
                    role,
                    utf8: UTF8Sequence(
                        scalar: value,
                        remaining: pending.remaining - 1,
                        nextByteRange: 0x80...0xBF,
                        scalarRange: pending.scalarRange))
                return .consumed
            }
            guard let scalar = Unicode.Scalar(value), let updated = roleByAppending(scalar, to: role) else {
                return .rejected
            }
            phase = .string(updated, utf8: nil)
            return .consumed
        }
        if byte == Self.quote { return closeString(role: role) }
        if byte == Self.backslash {
            phase = .stringEscape(role)
            return .consumed
        }
        if byte < 0x20 { return .rejected }  // raw control characters must be escaped
        if byte < 0x80 {
            guard let scalar = Unicode.Scalar(UInt32(byte)), let updated = roleByAppending(scalar, to: role) else {
                return .rejected
            }
            phase = .string(updated, utf8: nil)
            return .consumed
        }
        guard let sequence = Self.utf8Sequence(startingWith: byte), roleAllowsScalarRange(role, sequence.scalarRange) else {
            return .rejected
        }
        phase = .string(role, utf8: sequence)
        return .consumed
    }

    private mutating func closeString(role: StringRole) -> StepOutcome {
        switch role {
        case .plain:
            phase = valueCompletedPhase()
            return .consumed
        case .freeKey:
            phase = .expectColon(.anyJSON)
            return .consumed
        case .enumValue(let matcher):
            guard matcher.isComplete else { return .rejected }
            phase = valueCompletedPhase()
            return .consumed
        case .key(let matcher):
            guard let name = matcher.completedName, let property = markKeyUsed(name) else { return .rejected }
            phase = .expectColon(.schemaScalar(property))
            return .consumed
        }
    }

    private mutating func consumeEscapeByte(_ byte: UInt8, role: StringRole) -> StepOutcome {
        if byte == Self.quote { return appendDecodedScalar(0x22, role: role) }
        if byte == Self.backslash { return appendDecodedScalar(0x5C, role: role) }
        if byte == UInt8(ascii: "/") { return appendDecodedScalar(0x2F, role: role) }
        if byte == UInt8(ascii: "b") { return appendDecodedScalar(0x08, role: role) }
        if byte == UInt8(ascii: "f") { return appendDecodedScalar(0x0C, role: role) }
        if byte == UInt8(ascii: "n") { return appendDecodedScalar(0x0A, role: role) }
        if byte == UInt8(ascii: "r") { return appendDecodedScalar(0x0D, role: role) }
        if byte == UInt8(ascii: "t") { return appendDecodedScalar(0x09, role: role) }
        if byte == UInt8(ascii: "u") {
            phase = .unicodeEscape(role, nibbles: 0, value: 0)
            return .consumed
        }
        return .rejected
    }

    private mutating func consumeUnicodeEscapeByte(
        _ byte: UInt8, role: StringRole, nibbles: Int, value: UInt32
    ) -> StepOutcome {
        guard let nibble = Self.hexNibble(byte) else { return .rejected }
        let newValue = (value << 4) | UInt32(nibble)
        guard nibbles == 3 else {
            phase = .unicodeEscape(role, nibbles: nibbles + 1, value: newValue)
            return .consumed
        }
        if (0xD800...0xDBFF).contains(newValue) {
            // A high surrogate must be paired; only proceed when some candidate
            // could still complete with the pair's scalar.
            guard roleAllowsScalarRange(role, 0x10000...0x10FFFF) else { return .rejected }
            phase = .lowSurrogateEscape(role, high: newValue, offset: 0, value: 0)
            return .consumed
        }
        if (0xDC00...0xDFFF).contains(newValue) { return .rejected }  // lone low surrogate
        guard let scalar = Unicode.Scalar(newValue), let updated = roleByAppending(scalar, to: role) else {
            return .rejected
        }
        phase = .string(updated, utf8: nil)
        return .consumed
    }

    private mutating func consumeLowSurrogateByte(
        _ byte: UInt8, role: StringRole, high: UInt32, offset: Int, value: UInt32
    ) -> StepOutcome {
        switch offset {
        case 0:
            guard byte == Self.backslash else { return .rejected }
            phase = .lowSurrogateEscape(role, high: high, offset: 1, value: 0)
            return .consumed
        case 1:
            guard byte == UInt8(ascii: "u") else { return .rejected }
            phase = .lowSurrogateEscape(role, high: high, offset: 2, value: 0)
            return .consumed
        default:
            guard let nibble = Self.hexNibble(byte) else { return .rejected }
            let nibbleIndex = offset - 2
            if nibbleIndex == 0, nibble != 0xD { return .rejected }
            if nibbleIndex == 1, !(0xC...0xF).contains(nibble) { return .rejected }
            let newValue = (value << 4) | UInt32(nibble)
            guard nibbleIndex == 3 else {
                phase = .lowSurrogateEscape(role, high: high, offset: offset + 1, value: newValue)
                return .consumed
            }
            let scalarValue = 0x10000 + ((high - 0xD800) << 10) + (newValue - 0xDC00)
            guard let scalar = Unicode.Scalar(scalarValue), let updated = roleByAppending(scalar, to: role) else {
                return .rejected
            }
            phase = .string(updated, utf8: nil)
            return .consumed
        }
    }

    private mutating func appendDecodedScalar(_ value: UInt32, role: StringRole) -> StepOutcome {
        guard let scalar = Unicode.Scalar(value), let updated = roleByAppending(scalar, to: role) else {
            return .rejected
        }
        phase = .string(updated, utf8: nil)
        return .consumed
    }

    private func roleByAppending(_ scalar: Unicode.Scalar, to role: StringRole) -> StringRole? {
        switch role {
        case .plain:
            return .plain
        case .freeKey:
            return .freeKey
        case .key(var matcher):
            guard matcher.append(scalar) else { return nil }
            return .key(matcher)
        case .enumValue(var matcher):
            guard matcher.append(scalar) else { return nil }
            return .enumValue(matcher)
        }
    }

    private func roleAllowsScalarRange(_ role: StringRole, _ range: ClosedRange<UInt32>) -> Bool {
        switch role {
        case .plain, .freeKey:
            return true
        case .key(let matcher), .enumValue(let matcher):
            return matcher.canAppend(scalarIn: range)
        }
    }

    // MARK: Numbers and literals

    private mutating func consumeNumberByte(_ byte: UInt8, state: NumberState, kind: NumberKind) -> StepOutcome {
        switch state {
        case .sign:
            if byte == Self.zero {
                phase = .number(.zero, kind)
                return .consumed
            }
            if Self.isNonZeroDigit(byte) {
                phase = .number(.integerDigits, kind)
                return .consumed
            }
            return .rejected
        case .zero:
            if kind == .number {
                if byte == Self.dot {
                    phase = .number(.fractionStart, kind)
                    return .consumed
                }
                if byte == Self.eLower || byte == Self.eUpper {
                    phase = .number(.exponentStart, kind)
                    return .consumed
                }
            }
            return terminateNumber(byte, kind: kind)
        case .integerDigits:
            if Self.isDigit(byte) { return .consumed }
            if kind == .number {
                if byte == Self.dot {
                    phase = .number(.fractionStart, kind)
                    return .consumed
                }
                if byte == Self.eLower || byte == Self.eUpper {
                    phase = .number(.exponentStart, kind)
                    return .consumed
                }
            }
            return terminateNumber(byte, kind: kind)
        case .fractionStart:
            guard Self.isDigit(byte) else { return .rejected }
            phase = .number(.fractionDigits, kind)
            return .consumed
        case .fractionDigits:
            if Self.isDigit(byte) { return .consumed }
            if byte == Self.eLower || byte == Self.eUpper {
                phase = .number(.exponentStart, kind)
                return .consumed
            }
            return terminateNumber(byte, kind: kind)
        case .exponentStart:
            if byte == Self.plus || byte == Self.minus {
                phase = .number(.exponentSign, kind)
                return .consumed
            }
            if Self.isDigit(byte) {
                phase = .number(.exponentDigits, kind)
                return .consumed
            }
            return .rejected
        case .exponentSign:
            guard Self.isDigit(byte) else { return .rejected }
            phase = .number(.exponentDigits, kind)
            return .consumed
        case .exponentDigits:
            if Self.isDigit(byte) { return .consumed }
            return terminateNumber(byte, kind: kind)
        }
    }

    /// A terminable number ends here: reprocess the terminating byte in the
    /// next phase (`.complete` at the root, `.afterValue` inside a container).
    private mutating func terminateNumber(_ byte: UInt8, kind: NumberKind) -> StepOutcome {
        guard Self.isValueTerminator(byte) else { return .rejected }
        phase = valueCompletedPhase()
        return .retry
    }

    private mutating func consumeLiteralByte(_ byte: UInt8, literal: LiteralState) -> StepOutcome {
        if literal.matched < literal.expected.count {
            guard byte == literal.expected[literal.matched] else { return .rejected }
            var updated = literal
            updated.matched += 1
            phase = .literal(updated)
            return .consumed
        }
        guard Self.isValueTerminator(byte) else { return .rejected }
        phase = valueCompletedPhase()
        return .retry
    }

    // MARK: - Byte helpers

    private static let openBrace = UInt8(ascii: "{")
    private static let closeBrace = UInt8(ascii: "}")
    private static let openBracket = UInt8(ascii: "[")
    private static let closeBracket = UInt8(ascii: "]")
    private static let colon = UInt8(ascii: ":")
    private static let comma = UInt8(ascii: ",")
    private static let quote = UInt8(ascii: "\"")
    private static let backslash = UInt8(ascii: "\\")
    private static let minus = UInt8(ascii: "-")
    private static let plus = UInt8(ascii: "+")
    private static let zero = UInt8(ascii: "0")
    private static let dot = UInt8(ascii: ".")
    private static let eLower = UInt8(ascii: "e")
    private static let eUpper = UInt8(ascii: "E")

    private static let trueLiteral = Array("true".utf8)
    private static let falseLiteral = Array("false".utf8)
    private static let nullLiteral = Array("null".utf8)

    private static func isJSONWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
    }

    /// Bytes that may terminate a completed value.
    private static func isValueTerminator(_ byte: UInt8) -> Bool {
        isJSONWhitespace(byte) || byte == comma || byte == closeBrace || byte == closeBracket
    }

    private static func isDigit(_ byte: UInt8) -> Bool { byte >= 0x30 && byte <= 0x39 }

    private static func isNonZeroDigit(_ byte: UInt8) -> Bool { byte >= 0x31 && byte <= 0x39 }

    private static func hexNibble(_ byte: UInt8) -> Int? {
        if byte >= 0x30 && byte <= 0x39 { return Int(byte - 0x30) }
        if byte >= 0x41 && byte <= 0x46 { return Int(byte - 0x41 + 10) }
        if byte >= 0x61 && byte <= 0x66 { return Int(byte - 0x61 + 10) }
        return nil
    }

    /// Strict UTF-8 lead-byte table: rejects overlongs, surrogates, and
    /// scalars above U+10FFFF by constraining the continuation bytes.
    private static func utf8Sequence(startingWith lead: UInt8) -> UTF8Sequence? {
        switch lead {
        case 0xC2...0xDF:
            return UTF8Sequence(
                scalar: UInt32(lead & 0x1F), remaining: 1, nextByteRange: 0x80...0xBF, scalarRange: 0x80...0x7FF)
        case 0xE0:
            return UTF8Sequence(
                scalar: 0, remaining: 2, nextByteRange: 0xA0...0xBF, scalarRange: 0x800...0xFFF)
        case 0xE1...0xEC:
            return UTF8Sequence(
                scalar: UInt32(lead & 0x0F), remaining: 2, nextByteRange: 0x80...0xBF, scalarRange: 0x1000...0xCFFF)
        case 0xED:
            return UTF8Sequence(
                scalar: UInt32(lead & 0x0F), remaining: 2, nextByteRange: 0x80...0x9F, scalarRange: 0xD000...0xDFFF)
        case 0xEE...0xEF:
            return UTF8Sequence(
                scalar: UInt32(lead & 0x0F), remaining: 2, nextByteRange: 0x80...0xBF, scalarRange: 0xE000...0xFFFF)
        case 0xF0:
            return UTF8Sequence(
                scalar: 0, remaining: 3, nextByteRange: 0x90...0xBF, scalarRange: 0x10000...0x3FFFF)
        case 0xF1...0xF3:
            return UTF8Sequence(
                scalar: UInt32(lead & 0x07), remaining: 3, nextByteRange: 0x80...0xBF, scalarRange: 0x40000...0xFFFFF)
        case 0xF4:
            return UTF8Sequence(
                scalar: UInt32(lead & 0x07), remaining: 3, nextByteRange: 0x80...0x8F, scalarRange: 0x100000...0x10FFFF)
        default:
            return nil
        }
    }
}

// MARK: - Status and errors

/// Grammar status at the current position.
public enum JSONGrammarStatus: Sendable, Equatable {
    /// The root JSON value is not complete yet.
    case inProgress
    /// A complete root value was consumed; only JSON whitespace and
    /// end-of-sequence may follow.
    case complete
    /// End-of-sequence was consumed after completion.
    case finished
    /// An illegal byte was rejected; sticky until `reset()`.
    case failed
}

/// Typed constrained-decoding failures. Every case is fail-closed: none of
/// them may be translated into a successful structured response.
public enum JSONGrammarError: Error, Sendable, Equatable {
    /// The compiled format does not constrain generation (`.text`).
    case unconstrainedFormat
    /// A token id outside the supplied vocabulary was offered.
    case unknownToken(Int)
    /// A token fragment cannot be consumed by the current grammar state.
    case illegalToken(Int)
    /// End-of-sequence was offered before the root value was complete.
    case prematureEndOfSequence
    /// The grammar state already failed; no token can advance it.
    case grammarFailed
    /// Generation already finished with end-of-sequence.
    case alreadyFinished
    /// No token in the supplied vocabulary can advance the grammar.
    case noLegalContinuation
    /// The logits vocabulary does not match the fragment table.
    case vocabularyMismatch(expected: Int, actual: Int)
    /// Generation stopped with a reported `stop` before the root value was
    /// complete. Recorded by the engine's post-generation invariant, never by
    /// the processor itself (only the engine knows generation ended).
    case incompleteAtStop

    public var message: String {
        switch self {
        case .unconstrainedFormat:
            return "the compiled response format does not constrain generation"
        case .unknownToken(let tokenId):
            return "token id \(tokenId) is outside the supplied vocabulary"
        case .illegalToken(let tokenId):
            return "token id \(tokenId) cannot be consumed by the current grammar state"
        case .prematureEndOfSequence:
            return "end-of-sequence was reached before the JSON value was complete"
        case .grammarFailed:
            return "the grammar state already failed; no token can advance it"
        case .alreadyFinished:
            return "generation already finished with end-of-sequence"
        case .noLegalContinuation:
            return "no token in the vocabulary can advance the grammar"
        case .vocabularyMismatch(let expected, let actual):
            return "logits vocabulary size \(actual) does not match the token fragment table vocabulary size \(expected)"
        case .incompleteAtStop:
            return "generation stopped before a complete JSON value was produced"
        }
    }
}

extension JSONGrammarError: LocalizedError {
    public var errorDescription: String? { message }
}
