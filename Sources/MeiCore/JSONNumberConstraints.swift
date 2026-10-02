import Foundation

// MARK: - Exact decimal values

/// Exact base-10 decimal value used by the strict-schema numeric constraints.
///
/// The value is `sign · digits · 10^exponent` with the digits normalized: no
/// leading zeros, no trailing zeros, and an empty digit list is exactly zero.
/// Schema-side numbers (stored as `Double`) enter through their shortest
/// round-trip decimal form; generated literals are assembled from their exact
/// digits. Comparisons, equality, and divisibility are therefore exact —
/// `0.1 · 3 = 0.3` holds here where binary floating point cannot say so.
///
/// This type is deliberately model-free: it is shared by the compiler
/// (validation and canonical constraint keys), the grammar automaton (prefix
/// feasibility), and the diagnostic instance validator.
public struct DecimalLiteral: Equatable, Hashable, Sendable, Comparable {
    /// True for negative values; always false for zero.
    public let negative: Bool
    /// Significant decimal digits, most significant first.
    public let digits: [UInt8]
    /// Power of ten applied to `digits`.
    public let exponent: Int

    public static let zero = DecimalLiteral(negative: false, digits: [], exponent: 0)
    static let one = DecimalLiteral(negative: false, digits: [1], exponent: 0)

    public init(negative: Bool, digits: [UInt8], exponent: Int) {
        var digits = digits
        var exponent = exponent
        var firstNonZero = 0
        while firstNonZero < digits.count && digits[firstNonZero] == 0 { firstNonZero += 1 }
        if firstNonZero > 0 { digits.removeFirst(firstNonZero) }
        while let last = digits.last, last == 0 {
            digits.removeLast()
            exponent += 1
        }
        if digits.isEmpty {
            self.negative = false
            self.digits = []
            self.exponent = 0
        } else {
            self.negative = negative
            self.digits = digits
            self.exponent = exponent
        }
    }

    /// Parses a plain decimal literal such as `-12.5`, `1e+20`, `0.0`.
    public init?(parsing text: String) {
        let bytes = Array(text.utf8)
        var index = 0
        var negative = false
        if index < bytes.count, bytes[index] == UInt8(ascii: "-") {
            negative = true
            index += 1
        }
        var digits: [UInt8] = []
        var fractionCount = 0
        var sawDigit = false
        while index < bytes.count, bytes[index] >= 0x30, bytes[index] <= 0x39 {
            digits.append(bytes[index] - 0x30)
            sawDigit = true
            index += 1
        }
        if index < bytes.count, bytes[index] == UInt8(ascii: ".") {
            index += 1
            while index < bytes.count, bytes[index] >= 0x30, bytes[index] <= 0x39 {
                digits.append(bytes[index] - 0x30)
                fractionCount += 1
                sawDigit = true
                index += 1
            }
        }
        guard sawDigit else { return nil }
        var exponent = -fractionCount
        if index < bytes.count, bytes[index] == UInt8(ascii: "e") || bytes[index] == UInt8(ascii: "E") {
            index += 1
            var exponentNegative = false
            if index < bytes.count, bytes[index] == UInt8(ascii: "+") || bytes[index] == UInt8(ascii: "-") {
                exponentNegative = bytes[index] == UInt8(ascii: "-")
                index += 1
            }
            var exponentValue = 0
            var sawExponentDigit = false
            while index < bytes.count, bytes[index] >= 0x30, bytes[index] <= 0x39 {
                exponentValue = exponentValue * 10 + Int(bytes[index] - 0x30)
                if exponentValue > 1_000_000 { return nil }
                sawExponentDigit = true
                index += 1
            }
            guard sawExponentDigit else { return nil }
            exponent += exponentNegative ? -exponentValue : exponentValue
        }
        guard index == bytes.count else { return nil }
        self.init(negative: negative, digits: digits, exponent: exponent)
    }

    /// The exact decimal value of a finite `Double`, via its shortest
    /// round-trip decimal form (`0.1` stays `0.1`, not the binary expansion).
    public init?(double value: Double) {
        guard value.isFinite, let parsed = DecimalLiteral(parsing: String(value)) else { return nil }
        self = parsed
    }

    init(integer: Int) {
        var digits: [UInt8] = []
        var magnitude = UInt64(integer.magnitude)
        while magnitude > 0 {
            digits.insert(UInt8(magnitude % 10), at: 0)
            magnitude /= 10
        }
        self.init(negative: integer < 0, digits: digits, exponent: 0)
    }

    public var isZero: Bool { digits.isEmpty }

    /// Decadic exponent of the leading digit: value in `[10^e, 10^(e+1))`
    /// for positive values. Only meaningful when non-zero.
    var decadicExponent: Int { digits.count - 1 + exponent }

    /// True when the value is an integer (no fractional part).
    var isInteger: Bool { isZero || exponent >= 0 }

    func scaled(byPowerOf10 k: Int) -> DecimalLiteral {
        DecimalLiteral(negative: negative, digits: digits, exponent: exponent + k)
    }

    var magnitude: DecimalLiteral { DecimalLiteral(negative: false, digits: digits, exponent: exponent) }

    var negated: DecimalLiteral { DecimalLiteral(negative: !negative, digits: digits, exponent: exponent) }

    /// Deterministic, injective text form of the value: `"0"`, `"15e-1"`,
    /// `"-5e0"`, `"1e2"`. Used for canonical constraint keys and error text.
    var canonicalText: String {
        guard !isZero else { return "0" }
        let body = digits.map { String($0) }.joined()
        return (negative ? "-" : "") + body + "e" + String(exponent)
    }

    // MARK: Comparison

    public static func < (lhs: DecimalLiteral, rhs: DecimalLiteral) -> Bool {
        compare(lhs, rhs) < 0
    }

    static func compare(_ lhs: DecimalLiteral, _ rhs: DecimalLiteral) -> Int {
        if lhs.isZero && rhs.isZero { return 0 }
        if lhs.isZero { return rhs.negative ? 1 : -1 }
        if rhs.isZero { return lhs.negative ? -1 : 1 }
        if lhs.negative != rhs.negative { return lhs.negative ? -1 : 1 }
        var result = compareMagnitudes(lhs, rhs)
        if lhs.negative { result = -result }
        return result
    }

    /// Both operands non-zero; compares absolute values.
    static func compareMagnitudes(_ lhs: DecimalLiteral, _ rhs: DecimalLiteral) -> Int {
        let leftExponent = lhs.decadicExponent
        let rightExponent = rhs.decadicExponent
        if leftExponent != rightExponent { return leftExponent < rightExponent ? -1 : 1 }
        let count = Swift.max(lhs.digits.count, rhs.digits.count)
        for index in 0..<count {
            let left = index < lhs.digits.count ? lhs.digits[index] : 0
            let right = index < rhs.digits.count ? rhs.digits[index] : 0
            if left != right { return left < right ? -1 : 1 }
        }
        return 0
    }

    // MARK: Integer rounding

    /// Floor toward negative infinity, as an integer value.
    func floorInteger() -> DecimalLiteral {
        guard !isZero, exponent < 0 else { return self }
        let dropped = digits.suffix(-exponent)
        let kept = Array(digits.prefix(Swift.max(0, digits.count + exponent)))
        var result = DecimalLiteral(negative: false, digits: kept, exponent: 0)
        if dropped.contains(where: { $0 != 0 }), negative {
            result = DecimalLiteral.addingInteger(result, 1)
        }
        return negative ? result.negated : result
    }

    /// Ceiling toward positive infinity, as an integer value.
    func ceilInteger() -> DecimalLiteral {
        guard !isZero, exponent < 0 else { return self }
        let dropped = digits.suffix(-exponent)
        let kept = Array(digits.prefix(Swift.max(0, digits.count + exponent)))
        var result = DecimalLiteral(negative: false, digits: kept, exponent: 0)
        if dropped.contains(where: { $0 != 0 }), !negative {
            result = DecimalLiteral.addingInteger(result, 1)
        }
        return negative ? result.negated : result
    }

    /// Adds ±1 to an integer value (used for open-interval endpoint math).
    static func addingInteger(_ value: DecimalLiteral, _ delta: Int) -> DecimalLiteral {
        precondition(value.isInteger && (delta == 1 || delta == -1))
        if value.isZero {
            return DecimalLiteral(negative: delta < 0, digits: [1], exponent: 0)
        }
        var magnitudeDigits = value.digits + [UInt8](repeating: 0, count: value.exponent)
        let magnitudeDelta = value.negative ? -delta : delta
        if magnitudeDelta > 0 {
            var index = magnitudeDigits.count - 1
            while index >= 0 {
                if magnitudeDigits[index] == 9 {
                    magnitudeDigits[index] = 0
                    index -= 1
                } else {
                    magnitudeDigits[index] += 1
                    break
                }
            }
            if index < 0 { magnitudeDigits.insert(1, at: 0) }
        } else {
            var index = magnitudeDigits.count - 1
            while index >= 0 {
                if magnitudeDigits[index] == 0 {
                    magnitudeDigits[index] = 9
                    index -= 1
                } else {
                    magnitudeDigits[index] -= 1
                    break
                }
            }
        }
        return DecimalLiteral(negative: value.negative, digits: magnitudeDigits, exponent: 0)
    }

    static func subtractingInteger(_ value: DecimalLiteral, _ delta: Int) -> DecimalLiteral {
        addingInteger(value, -delta)
    }

    // MARK: Addition (non-negative operands only)

    /// Exact sum of two non-negative values.
    static func addMagnitudes(_ lhs: DecimalLiteral, _ rhs: DecimalLiteral) -> DecimalLiteral {
        precondition(!lhs.negative && !rhs.negative)
        let common = Swift.min(lhs.exponent, rhs.exponent)
        var left = lhs.digits + [UInt8](repeating: 0, count: lhs.exponent - common)
        var right = rhs.digits + [UInt8](repeating: 0, count: rhs.exponent - common)
        if left.count < right.count {
            left = [UInt8](repeating: 0, count: right.count - left.count) + left
        } else if right.count < left.count {
            right = [UInt8](repeating: 0, count: left.count - right.count) + right
        }
        var sum = [UInt8](repeating: 0, count: left.count + 1)
        var carry = 0
        for index in stride(from: left.count - 1, through: 0, by: -1) {
            let total = Int(left[index]) + Int(right[index]) + carry
            sum[index + 1] = UInt8(total % 10)
            carry = total / 10
        }
        sum[0] = UInt8(carry)
        return DecimalLiteral(negative: false, digits: sum, exponent: common)
    }

    // MARK: Divisibility

    /// True when `self / divisor` is an integer. `divisor` must be non-zero;
    /// its sign is irrelevant. Divisor significands are at most 17 digits
    /// (shortest `Double` forms), so 64-bit streaming arithmetic is exact.
    func isMultiple(of divisor: DecimalLiteral) -> Bool {
        precondition(!divisor.isZero)
        if isZero { return true }
        guard let divisorValue = DecimalLiteral.uint64Value(divisor.digits) else { return false }
        let delta = exponent - divisor.exponent
        if delta >= 0 {
            let remainder = DecimalLiteral.modStreaming(digits, modulus: divisorValue)
            if remainder == 0 { return true }
            let factor = DecimalLiteral.powMod(10, delta, divisorValue)
            return DecimalLiteral.mulMod(remainder, factor, divisorValue) == 0
        }
        let shift = -delta
        guard digits.count >= shift else { return false }
        for offset in 0..<shift where digits[digits.count - 1 - offset] != 0 { return false }
        let stripped = Array(digits.prefix(digits.count - shift))
        return DecimalLiteral.modStreaming(stripped, modulus: divisorValue) == 0
    }

    /// When `self == base · 10^E` for an integer `E`, returns `E`; otherwise
    /// nil. Zero matches only zero (with `E == 0`).
    func powerOfTenRatio(to base: DecimalLiteral) -> Int? {
        if isZero || base.isZero { return isZero && base.isZero ? 0 : nil }
        guard negative == base.negative, digits == base.digits else { return nil }
        return exponent - base.exponent
    }

    /// The least positive integer multiple of `self` (which must be positive).
    ///
    /// An integer value is a multiple of `m = D · 10^e` exactly when it is a
    /// multiple of `D / gcd(D, 10^-e)` for `e < 0` (and of `m` itself for
    /// `e ≥ 0`): `D / gcd` is the smallest `k · m` that lands on an integer.
    /// This is what makes integer satisfiability and integer prefix
    /// feasibility exact for fractional `multipleOf` (e.g. `1.5` becomes `3`,
    /// `0.25` becomes `1`).
    ///
    /// Returns nil when the significand exceeds 64-bit arithmetic; compiler
    /// inputs are shortest `Double` forms (≤ 17 digits) and always fit.
    func leastPositiveIntegerMultiple() -> DecimalLiteral? {
        precondition(!isZero && !negative)
        if exponent >= 0 { return self }
        guard let significand = DecimalLiteral.uint64Value(digits) else { return nil }
        let shift = -exponent
        // gcd(D, 10^shift) = 2^min(v2(D), shift) · 5^min(v5(D), shift).
        var twos = 0
        var value = significand
        while twos < shift, value % 2 == 0 {
            value /= 2
            twos += 1
        }
        var fives = 0
        value = significand
        while fives < shift, value % 5 == 0 {
            value /= 5
            fives += 1
        }
        var divisor: UInt64 = 1
        for _ in 0..<twos { divisor *= 2 }
        for _ in 0..<fives { divisor *= 5 }
        return DecimalLiteral(parsing: String(significand / divisor))
    }

    // MARK: Integer division of exact decimals

    /// floor(numerator / denominator), denominator > 0, as an integer value.
    static func floorDivide(_ numerator: DecimalLiteral, by denominator: DecimalLiteral) -> DecimalLiteral {
        let (quotient, exact) = floorDivideMagnitudes(numerator.magnitude, denominator)
        if numerator.negative {
            return exact ? quotient.negated : subtractingInteger(quotient.negated, 1)
        }
        return quotient
    }

    /// ceil(numerator / denominator), denominator > 0, as an integer value.
    static func ceilDivide(_ numerator: DecimalLiteral, by denominator: DecimalLiteral) -> DecimalLiteral {
        let (quotient, exact) = floorDivideMagnitudes(numerator.magnitude, denominator)
        if numerator.negative {
            return quotient.negated
        }
        return exact ? quotient : addingInteger(quotient, 1)
    }

    /// floor(|numerator| / denominator) with exactness flag; both magnitudes,
    /// denominator non-zero positive.
    static func floorDivideMagnitudes(
        _ numerator: DecimalLiteral, _ denominator: DecimalLiteral
    ) -> (quotient: DecimalLiteral, exact: Bool) {
        guard let divisor = uint64Value(denominator.digits) else {
            return (.zero, false)
        }
        let delta = numerator.exponent - denominator.exponent
        if delta >= 0 {
            let scaledDigits = numerator.digits + [UInt8](repeating: 0, count: delta)
            let (quotientDigits, remainder) = divideStreaming(scaledDigits, divisor: divisor)
            return (DecimalLiteral(negative: false, digits: quotientDigits, exponent: 0), remainder == 0)
        }
        let shift = -delta
        let (quotientDigits, remainder) = divideStreaming(numerator.digits, divisor: divisor)
        let dropped = quotientDigits.suffix(shift)
        let kept = shift < quotientDigits.count
            ? Array(quotientDigits.prefix(quotientDigits.count - shift)) : []
        let exact = remainder == 0 && dropped.allSatisfy { $0 == 0 }
        return (DecimalLiteral(negative: false, digits: kept, exponent: 0), exact)
    }

    // MARK: Digit-string arithmetic

    static func uint64Value(_ digits: [UInt8]) -> UInt64? {
        var value: UInt64 = 0
        for digit in digits {
            let (multiplied, overflow1) = value.multipliedReportingOverflow(by: 10)
            guard !overflow1 else { return nil }
            let (added, overflow2) = multiplied.addingReportingOverflow(UInt64(digit))
            guard !overflow2 else { return nil }
            value = added
        }
        return value
    }

    static func modStreaming(_ digits: [UInt8], modulus: UInt64) -> UInt64 {
        precondition(modulus > 0)
        var remainder: UInt64 = 0
        for digit in digits {
            remainder = (remainder * 10 + UInt64(digit)) % modulus
        }
        return remainder
    }

    static func divideStreaming(_ digits: [UInt8], divisor: UInt64) -> ([UInt8], UInt64) {
        var quotient: [UInt8] = []
        quotient.reserveCapacity(digits.count)
        var remainder: UInt64 = 0
        for digit in digits {
            let accumulator = remainder * 10 + UInt64(digit)
            quotient.append(UInt8(accumulator / divisor))
            remainder = accumulator % divisor
        }
        return (quotient, remainder)
    }

    static func mulMod(_ lhs: UInt64, _ rhs: UInt64, _ modulus: UInt64) -> UInt64 {
        precondition(modulus > 0 && lhs < modulus && rhs < modulus)
        let product = lhs.multipliedFullWidth(by: rhs)
        return modulus.dividingFullWidth(product).remainder
    }

    static func powMod(_ base: UInt64, _ exponent: Int, _ modulus: UInt64) -> UInt64 {
        precondition(exponent >= 0 && modulus > 0)
        var result: UInt64 = 1 % modulus
        var factor = base % modulus
        var remaining = exponent
        while remaining > 0 {
            if remaining & 1 == 1 { result = mulMod(result, factor, modulus) }
            factor = mulMod(factor, factor, modulus)
            remaining >>= 1
        }
        return result
    }

    // MARK: Interval helpers

    /// True when some integer multiple of `multiple` lies within the interval.
    /// A nil bound is unbounded on that side.
    static func intervalContainsMultiple(
        lower: (value: DecimalLiteral, inclusive: Bool)?,
        upper: (value: DecimalLiteral, inclusive: Bool)?,
        multiple: DecimalLiteral
    ) -> Bool {
        precondition(!multiple.isZero && !multiple.negative)
        let kMin: DecimalLiteral?
        if let lower {
            kMin = lower.inclusive
                ? ceilDivide(lower.value, by: multiple)
                : addingInteger(floorDivide(lower.value, by: multiple), 1)
        } else {
            kMin = nil
        }
        let kMax: DecimalLiteral?
        if let upper {
            kMax = upper.inclusive
                ? floorDivide(upper.value, by: multiple)
                : subtractingInteger(ceilDivide(upper.value, by: multiple), 1)
        } else {
            kMax = nil
        }
        switch (kMin, kMax) {
        case (nil, _), (_, nil): return true
        case (let low?, let high?): return low <= high
        }
    }

    /// True when the interval contains at least one value.
    static func intervalNonempty(
        lower: (value: DecimalLiteral, inclusive: Bool)?,
        upper: (value: DecimalLiteral, inclusive: Bool)?
    ) -> Bool {
        guard let lower, let upper else { return true }
        let order = compare(lower.value, upper.value)
        if order < 0 { return true }
        if order > 0 { return false }
        return lower.inclusive && upper.inclusive
    }

    /// Largest `e` with `base · 10^e ≤ limit` (or `<` when `strict`), for
    /// `base > 0`. Returns nil when no exponent satisfies it.
    static func maxExponentScaling(_ base: DecimalLiteral, atMost limit: DecimalLiteral, strict: Bool) -> Int? {
        precondition(!base.isZero && !base.negative)
        guard !limit.negative, !limit.isZero else { return nil }
        let candidate = limit.decadicExponent - base.decadicExponent
        let comparison = compareMagnitudes(base.scaled(byPowerOf10: candidate), limit)
        let satisfies = strict ? comparison < 0 : comparison <= 0
        return satisfies ? candidate : candidate - 1
    }

    /// Smallest `e` with `base · 10^e ≥ limit` (or `>` when `strict`), for
    /// `base > 0`. Returns nil when every exponent satisfies it (the lower
    /// limit is vacuous).
    static func minExponentScaling(_ base: DecimalLiteral, atLeast limit: DecimalLiteral, strict: Bool) -> Int? {
        precondition(!base.isZero && !base.negative)
        if limit.negative || limit.isZero { return nil }
        let candidate = limit.decadicExponent - base.decadicExponent
        let comparison = compareMagnitudes(base.scaled(byPowerOf10: candidate), limit)
        let satisfies = strict ? comparison > 0 : comparison >= 0
        return satisfies ? candidate : candidate + 1
    }

    /// Canonical decimal digits of a non-negative integer value, no leading
    /// zeros (`0` yields `[0]`).
    static func canonicalDigits(ofInteger value: DecimalLiteral) -> [UInt8] {
        precondition(value.isInteger && !value.negative)
        if value.isZero { return [0] }
        return value.digits + [UInt8](repeating: 0, count: value.exponent)
    }
}

// MARK: - Compiled numeric constraints

/// Numeric constraint keywords for `number`/`integer` scalars, exactly as
/// declared by the schema (all present constraints apply; `minimum` and
/// `exclusiveMinimum` may coexist and intersect).
public struct NumericConstraints: Equatable, Sendable {
    public let minimum: DecimalLiteral?
    public let maximum: DecimalLiteral?
    public let exclusiveMinimum: DecimalLiteral?
    public let exclusiveMaximum: DecimalLiteral?
    /// Strictly positive.
    public let multipleOf: DecimalLiteral?

    public init(
        minimum: DecimalLiteral? = nil,
        maximum: DecimalLiteral? = nil,
        exclusiveMinimum: DecimalLiteral? = nil,
        exclusiveMaximum: DecimalLiteral? = nil,
        multipleOf: DecimalLiteral? = nil
    ) {
        self.minimum = minimum
        self.maximum = maximum
        self.exclusiveMinimum = exclusiveMinimum
        self.exclusiveMaximum = exclusiveMaximum
        self.multipleOf = multipleOf
    }

    /// True when no keyword is set.
    public var isEmpty: Bool {
        minimum == nil && maximum == nil && exclusiveMinimum == nil && exclusiveMaximum == nil && multipleOf == nil
    }
}

// MARK: - Prefix feasibility (the constrained number grammar)

/// A prefix of a JSON number literal as tracked during generation. The
/// grammar fills this in byte by byte; the feasibility engine decides whether
/// some completion of the prefix is accepted by the numeric constraints.
struct NumberPrefix: Equatable {
    enum ExponentPrefix: Equatable {
        /// `e` — a sign and digits are still open (any integer exponent).
        case bare
        /// `e+` / `e-` — digits still open.
        case sign(negative: Bool)
        /// `e5`, `e-05`, ...
        case digits(negative: Bool, digits: [UInt8])
    }

    var negative = false
    var intDigits: [UInt8] = []
    var hasFraction = false
    var fractionDigits: [UInt8] = []
    var exponent: ExponentPrefix? = nil
}

/// Everything the number grammar must enforce for one scalar field: bounds,
/// `multipleOf`, and the enum values (for `number`/`integer` fields).
struct NumericConstraintSet: Equatable {
    let bounds: NumericConstraints?
    let allowedValues: [DecimalLiteral]?

    var isEmpty: Bool { (bounds?.isEmpty ?? true) && allowedValues == nil }

    // MARK: Exact acceptance

    /// The exact completed value satisfies every declared constraint.
    func accepts(_ value: DecimalLiteral) -> Bool {
        if let bounds {
            if let minimum = bounds.minimum, !(value >= minimum) { return false }
            if let maximum = bounds.maximum, !(value <= maximum) { return false }
            if let exclusiveMinimum = bounds.exclusiveMinimum, !(value > exclusiveMinimum) { return false }
            if let exclusiveMaximum = bounds.exclusiveMaximum, !(value < exclusiveMaximum) { return false }
            if let multipleOf = bounds.multipleOf, !value.isMultiple(of: multipleOf) { return false }
        }
        if let allowedValues, !allowedValues.contains(value) { return false }
        return true
    }

    // MARK: Prefix feasibility

    /// True when some completion of `prefix` is accepted. Exact: it decides
    /// over the full JSON number space (integer digits, fractions, and
    /// exponents) using exact decimal arithmetic.
    func hasCompletion(prefix: NumberPrefix, kind: CompiledJSONSchema.ScalarType) -> Bool {
        switch kind {
        case .integer:
            return integerHasCompletion(prefix)
        case .number:
            return numberHasCompletion(prefix)
        case .string, .boolean:
            return true
        }
    }

    /// Some literal of `kind` is accepted (compile-time satisfiability).
    func isSatisfiable(kind: CompiledJSONSchema.ScalarType) -> Bool {
        if let allowedValues {
            return allowedValues.contains { accepts($0) }
        }
        switch kind {
        case .integer:
            guard let interval = integerInterval() else { return false }
            if let multipleOf = bounds?.multipleOf {
                // Integer values that are multiples of a fractional
                // `multipleOf` are exactly the multiples of its least positive
                // integer multiple (e.g. 1.5 -> 3: 1..2 has none, 1..4 has 3).
                guard let integerMultiple = multipleOf.leastPositiveIntegerMultiple() else { return false }
                return DecimalLiteral.intervalContainsMultiple(
                    lower: interval.lower.map { ($0, true) },
                    upper: interval.upper.map { ($0, true) },
                    multiple: integerMultiple)
            }
            return true
        case .number:
            let (lower, upper) = effectiveBounds()
            guard DecimalLiteral.intervalNonempty(lower: lower, upper: upper) else { return false }
            if let multipleOf = bounds?.multipleOf {
                return DecimalLiteral.intervalContainsMultiple(lower: lower, upper: upper, multiple: multipleOf)
            }
            return true
        case .string, .boolean:
            return true
        }
    }

    // MARK: Integer feasibility

    /// Integer literals in the current grammar are digit-only (no fraction,
    /// no exponent), so the reachable set from a prefix is a digit-prefix
    /// window union: `[x·10^j, (x+1)·10^j)` for every `j ≥ 0` (plus the exact
    /// value `0` for a leading zero).
    ///
    /// Only the windows that can meet both bounds are considered, and their
    /// exponent range is computed directly from the bounds
    /// (`minExponentScaling`/`maxExponentScaling`) instead of walking the
    /// whole decade gap one `j` at a time with growing big-integer
    /// arithmetic — the walk dominated prefix checks for 1e18..1e300-scale
    /// bounds. With a `multipleOf`, the divisibility target is the least
    /// positive *integer* multiple, and windows narrower than it are checked
    /// explicitly while unclipped windows at least that wide are guaranteed
    /// to contain one.
    private func integerHasCompletion(_ prefix: NumberPrefix) -> Bool {
        precondition(prefix.exponent == nil && !prefix.hasFraction)
        if let allowedValues {
            return allowedValues.contains { candidate in
                accepts(candidate) && integerReaches(candidate, prefix: prefix)
            }
        }
        if prefix.intDigits == [0] {
            // `0` cannot be extended at all in integer syntax.
            return accepts(.zero)
        }
        let interval = integerInterval()
        guard let interval else { return false }
        // `-` (no digits) can complete to any non-positive integer.
        let start = prefix.intDigits.isEmpty
            ? DecimalLiteral.zero
            : DecimalLiteral(negative: false, digits: prefix.intDigits, exponent: 0)
        let lower = prefix.negative ? interval.upper?.negated : interval.lower
        let upper = prefix.negative ? interval.lower?.negated : interval.upper
        let multiple: DecimalLiteral?
        if let multipleOf = bounds?.multipleOf {
            // Integer multiples of a fractional `multipleOf` are exactly the
            // multiples of its least positive integer multiple (1.5 -> 3).
            guard let adjusted = multipleOf.leastPositiveIntegerMultiple() else { return false }
            multiple = adjusted
        } else {
            multiple = nil
        }
        if start.isZero {
            // The magnitude reachable set is [0, ∞): 0, 1, 2, ... all occur.
            let clippedLower: (value: DecimalLiteral, inclusive: Bool)
            if let lower, lower > .zero {
                clippedLower = (lower, true)
            } else {
                clippedLower = (.zero, true)
            }
            if let multiple {
                return DecimalLiteral.intervalContainsMultiple(
                    lower: clippedLower, upper: upper.map { ($0, true) }, multiple: multiple)
            }
            return DecimalLiteral.intervalNonempty(lower: clippedLower, upper: upper.map { ($0, true) })
        }
        // The window at j = 0 starts at `x` and every larger window starts
        // even higher, so an already-above-range prefix can never come back.
        if let upper, start > upper { return false }
        // Window `j` intersects [lower, upper] only when
        // `(x+1)·10^j > lower` and `x·10^j ≤ upper`. The first such exponent
        // is computed directly; the last is the largest scaling that keeps
        // the window start at or below the upper bound.
        let top = DecimalLiteral.addMagnitudes(start, .one)
        var jStart = 0
        if let lower, lower > .zero, start < lower {
            if let first = DecimalLiteral.minExponentScaling(top, atLeast: lower, strict: true) {
                jStart = Swift.max(0, first)
            }
        }
        var jEnd: Int? = nil
        if let upper {
            // `start > upper` was rejected above, so the window at j = 0 fits
            // under the upper bound; it is also the last one exactly when one
            // more decade would overshoot.
            if start.scaled(byPowerOf10: 1) > upper {
                jEnd = 0
            } else {
                guard let last = DecimalLiteral.maxExponentScaling(start, atMost: upper, strict: false) else {
                    return false
                }
                jEnd = last
            }
        }
        if let jEnd, jStart > jEnd { return false }
        guard let multiple else {
            // The window at `jStart` (or any larger one) always contains an
            // integer within the bounds.
            return true
        }
        // Below this exponent the whole window sits at or below the least
        // positive multiple, so it cannot contain one; skip those exponents
        // (only possible while the j = 0 window top is still at or below it).
        if top <= multiple,
            let first = DecimalLiteral.minExponentScaling(top, atLeast: multiple, strict: true) {
            jStart = Swift.max(jStart, first)
        }
        var j = jStart
        while true {
            if let jEnd, j > jEnd { return false }
            let windowLower = start.scaled(byPowerOf10: j)
            let exclusiveWindowTop = DecimalLiteral.addingInteger(top.scaled(byPowerOf10: j), -1)
            let clippedLower = lower.map { Swift.max($0, windowLower) } ?? windowLower
            let clippedUpper = upper.map { Swift.min($0, exclusiveWindowTop) } ?? exclusiveWindowTop
            // An unclipped window holding at least `multiple` consecutive
            // integers is guaranteed to contain a multiple: stop scanning.
            if clippedLower == windowLower, clippedUpper == exclusiveWindowTop,
                DecimalLiteral.compareMagnitudes(DecimalLiteral.one.scaled(byPowerOf10: j), multiple) >= 0 {
                return true
            }
            if DecimalLiteral.intervalContainsMultiple(
                lower: (clippedLower, true), upper: (clippedUpper, true), multiple: multiple) {
                return true
            }
            j += 1
        }
    }

    /// Inclusive integer bounds from the declared constraints, in value
    /// space. Returns nil when no integer satisfies the bounds.
    private func integerInterval() -> (lower: DecimalLiteral?, upper: DecimalLiteral?)? {
        var lower: DecimalLiteral? = nil
        var upper: DecimalLiteral? = nil
        if let bounds {
            if let minimum = bounds.minimum {
                let candidate = minimum.ceilInteger()
                lower = lower.map { Swift.max($0, candidate) } ?? candidate
            }
            if let exclusiveMinimum = bounds.exclusiveMinimum {
                let candidate = DecimalLiteral.addingInteger(exclusiveMinimum.floorInteger(), 1)
                lower = lower.map { Swift.max($0, candidate) } ?? candidate
            }
            if let maximum = bounds.maximum {
                let candidate = maximum.floorInteger()
                upper = upper.map { Swift.min($0, candidate) } ?? candidate
            }
            if let exclusiveMaximum = bounds.exclusiveMaximum {
                let candidate = DecimalLiteral.addingInteger(exclusiveMaximum.ceilInteger(), -1)
                upper = upper.map { Swift.min($0, candidate) } ?? candidate
            }
        }
        if let lower, let upper, lower > upper { return nil }
        return (lower, upper)
    }

    /// Whether `candidate` (an integer value) lies in the digit-prefix window
    /// union of the prefix.
    private func integerReaches(_ candidate: DecimalLiteral, prefix: NumberPrefix) -> Bool {
        if candidate.isZero {
            return prefix.intDigits.isEmpty || prefix.intDigits == [0]
        }
        guard candidate.negative == prefix.negative else { return false }
        if prefix.intDigits.isEmpty { return true }
        let candidateDigits = DecimalLiteral.canonicalDigits(ofInteger: candidate.magnitude)
        let prefixDigits = prefix.intDigits == [0] ? [0] : prefix.intDigits
        guard candidateDigits.count >= prefixDigits.count else { return false }
        return Array(candidateDigits.prefix(prefixDigits.count)) == prefixDigits
    }

    // MARK: Number feasibility (no exponent yet)

    private func numberHasCompletion(_ prefix: NumberPrefix) -> Bool {
        if let exponent = prefix.exponent {
            return numberExponentRegime(prefix, exponent: exponent)
        }
        let (a, w) = numberMagnitudeInterval(prefix)
        if let allowedValues {
            return allowedValues.contains { candidate in
                accepts(candidate) && numberRegimeOneReaches(candidate, prefix: prefix, a: a, w: w)
            }
        }
        let interval = magnitudeInterval(negative: prefix.negative)
        if a.isZero {
            // The magnitude reachable set is [0, ∞).
            let clippedLower: (value: DecimalLiteral, inclusive: Bool)?
            if let lower = interval.lower {
                let order = DecimalLiteral.compare(lower.value, .zero)
                if order > 0 {
                    clippedLower = lower
                } else if order == 0 {
                    clippedLower = (.zero, lower.inclusive)
                } else {
                    clippedLower = (.zero, true)
                }
            } else {
                clippedLower = (.zero, true)
            }
            if let multipleOf = bounds?.multipleOf {
                return DecimalLiteral.intervalContainsMultiple(
                    lower: clippedLower, upper: interval.upper, multiple: multipleOf)
            }
            return DecimalLiteral.intervalNonempty(lower: clippedLower, upper: interval.upper)
        }
        if let multipleOf = bounds?.multipleOf {
            return scaledIntervalContainsMultiple(a: a, w: w, interval: interval, multiple: multipleOf)
        }
        return scaledIntervalIntersects(a: a, w: w, interval: interval)
    }

    /// `(a, w)`: magnitudes reachable without an exponent are
    /// `{x · 10^e : x ∈ [a, a+w)}` for every integer `e`.
    private func numberMagnitudeInterval(_ prefix: NumberPrefix) -> (DecimalLiteral, DecimalLiteral) {
        let combined = prefix.intDigits + prefix.fractionDigits
        let a = DecimalLiteral(negative: false, digits: combined, exponent: -prefix.fractionDigits.count)
        let w = DecimalLiteral(negative: false, digits: [1], exponent: -prefix.fractionDigits.count)
        return (a, w)
    }

    /// Effective bounds in magnitude space after applying the prefix sign.
    private func magnitudeInterval(
        negative: Bool
    ) -> (lower: (value: DecimalLiteral, inclusive: Bool)?, upper: (value: DecimalLiteral, inclusive: Bool)?) {
        let (lower, upper) = effectiveBounds()
        guard negative else { return (lower, upper) }
        return (
            upper.map { ($0.value.negated, $0.inclusive) },
            lower.map { ($0.value.negated, $0.inclusive) }
        )
    }

    /// Intersection of the declared bounds as one interval (value space).
    func effectiveBounds() -> (
        lower: (value: DecimalLiteral, inclusive: Bool)?, upper: (value: DecimalLiteral, inclusive: Bool)?
    ) {
        var lower: (value: DecimalLiteral, inclusive: Bool)? = nil
        var upper: (value: DecimalLiteral, inclusive: Bool)? = nil
        if let bounds {
            if let minimum = bounds.minimum { lower = (minimum, true) }
            if let exclusiveMinimum = bounds.exclusiveMinimum {
                if let current = lower {
                    let order = DecimalLiteral.compare(exclusiveMinimum, current.value)
                    if order > 0 || (order == 0 && current.inclusive) { lower = (exclusiveMinimum, false) }
                } else {
                    lower = (exclusiveMinimum, false)
                }
            }
            if let maximum = bounds.maximum { upper = (maximum, true) }
            if let exclusiveMaximum = bounds.exclusiveMaximum {
                if let current = upper {
                    let order = DecimalLiteral.compare(exclusiveMaximum, current.value)
                    if order < 0 || (order == 0 && current.inclusive) { upper = (exclusiveMaximum, false) }
                } else {
                    upper = (exclusiveMaximum, false)
                }
            }
        }
        return (lower, upper)
    }

    /// `∃ e ∈ ℤ : [a·10^e, (a+w)·10^e) ∩ interval ≠ ∅`, for `a > 0`.
    private func scaledIntervalIntersects(
        a: DecimalLiteral,
        w: DecimalLiteral,
        interval: (lower: (value: DecimalLiteral, inclusive: Bool)?, upper: (value: DecimalLiteral, inclusive: Bool)?)
    ) -> Bool {
        let top = DecimalLiteral.addMagnitudes(a, w)
        var eMax: Int? = nil
        if let upper = interval.upper {
            guard upper.value > .zero else { return false }
            guard let limit = DecimalLiteral.maxExponentScaling(a, atMost: upper.value, strict: !upper.inclusive) else {
                return false
            }
            eMax = limit
        }
        var eMin: Int? = nil
        if let lower = interval.lower {
            // The half-open window must strictly exceed the lower bound to
            // overlap it, regardless of the bound's own inclusivity.
            eMin = DecimalLiteral.minExponentScaling(top, atLeast: lower.value, strict: true)
        }
        switch (eMin, eMax) {
        case (nil, _), (_, nil): return true
        case (let low?, let high?): return low <= high
        }
    }

    /// `∃ e ∈ ℤ` such that the scaled half-open window contains a multiple.
    private func scaledIntervalContainsMultiple(
        a: DecimalLiteral,
        w: DecimalLiteral,
        interval: (lower: (value: DecimalLiteral, inclusive: Bool)?, upper: (value: DecimalLiteral, inclusive: Bool)?),
        multiple: DecimalLiteral
    ) -> Bool {
        let top = DecimalLiteral.addMagnitudes(a, w)
        guard let upper = interval.upper else {
            // No upper bound: a large enough window is wider than the spacing
            // and above any finite lower bound, so a multiple always exists.
            return true
        }
        guard upper.value > .zero else { return false }
        guard let eMax = DecimalLiteral.maxExponentScaling(a, atMost: upper.value, strict: !upper.inclusive) else {
            return false
        }
        var eMin: Int? = nil
        if let lower = interval.lower {
            eMin = DecimalLiteral.minExponentScaling(top, atLeast: lower.value, strict: true)
        }
        // Below this exponent the window sits inside (0, m) and can never
        // contain a non-zero multiple (0 is not in the half-open window).
        let eFloor = multiple.decadicExponent - a.decadicExponent - 1
        var e = Swift.max(eMin ?? eFloor, eFloor)
        var iterations = 0
        while iterations < 800 {
            if e > eMax { return false }
            let lowerValue = a.scaled(byPowerOf10: e)
            let upperValue = top.scaled(byPowerOf10: e)
            var clippedLower: (value: DecimalLiteral, inclusive: Bool) = (lowerValue, true)
            if let lower = interval.lower {
                let order = DecimalLiteral.compare(lower.value, lowerValue)
                if order > 0 {
                    clippedLower = lower
                } else if order == 0 {
                    clippedLower = (lowerValue, lower.inclusive)
                }
            }
            var clippedUpper: (value: DecimalLiteral, inclusive: Bool) = (upperValue, false)
            if let upperBound = interval.upper {
                let order = DecimalLiteral.compare(upperBound.value, upperValue)
                if order < 0 {
                    clippedUpper = upperBound
                } else if order == 0 {
                    clippedUpper = (upperValue, false)
                }
            }
            if DecimalLiteral.intervalContainsMultiple(
                lower: clippedLower, upper: clippedUpper, multiple: multiple) {
                return true
            }
            // An unclipped window of width ≥ the multiple spacing is
            // guaranteed to contain a multiple: stop scanning.
            if clippedLower.value == lowerValue, clippedUpper.value == upperValue,
                DecimalLiteral.compareMagnitudes(w.scaled(byPowerOf10: e), multiple) >= 0 {
                return true
            }
            e += 1
            iterations += 1
        }
        return false
    }

    /// Exact reachability of a candidate value in regime 1.
    private func numberRegimeOneReaches(
        _ candidate: DecimalLiteral, prefix: NumberPrefix, a: DecimalLiteral, w: DecimalLiteral
    ) -> Bool {
        if candidate.isZero { return a.isZero }
        guard candidate.negative == prefix.negative else { return false }
        if a.isZero { return true }
        let magnitude = candidate.magnitude
        // v is reachable iff v·10^-e lands in [a, a+w) for some integer e;
        // the aligning exponent puts v·10^-e in a's decade.
        let base = magnitude.decadicExponent - a.decadicExponent
        let top = DecimalLiteral.addMagnitudes(a, w)
        for e in [base - 1, base, base + 1] {
            let scaled = magnitude.scaled(byPowerOf10: -e)
            if scaled >= a && scaled < top { return true }
        }
        return false
    }

    // MARK: Number feasibility (exponent seen)

    private func numberExponentRegime(_ prefix: NumberPrefix, exponent: NumberPrefix.ExponentPrefix) -> Bool {
        let combined = prefix.intDigits + prefix.fractionDigits
        let a0 = DecimalLiteral(
            negative: false, digits: combined, exponent: -prefix.fractionDigits.count)
        if a0.isZero { return accepts(.zero) }
        let reachability = ExponentReachability(prefix: exponent)
        if let allowedValues {
            return allowedValues.contains { candidate in
                guard accepts(candidate) else { return false }
                // Value = ±a0·10^E: the candidate must match the prefix sign
                // and never be zero (a0 is non-zero here).
                guard !candidate.isZero, candidate.negative == prefix.negative else { return false }
                guard let e = candidate.magnitude.powerOfTenRatio(to: a0) else { return false }
                return reachability.contains(e)
            }
        }
        // Bounds give an integer exponent range; multipleOf a lower threshold.
        let (lower, upper) = magnitudeInterval(negative: prefix.negative)
        var eMin: Int? = nil
        var eMax: Int? = nil
        if let upper = upper {
            guard upper.value > .zero else { return false }
            guard let limit = DecimalLiteral.maxExponentScaling(a0, atMost: upper.value, strict: !upper.inclusive) else {
                return false
            }
            eMax = limit
        }
        if let lower = lower {
            eMin = DecimalLiteral.minExponentScaling(a0, atLeast: lower.value, strict: !lower.inclusive)
        }
        if let multipleOf = bounds?.multipleOf {
            guard let threshold = multipleExponentThreshold(a0: a0, multiple: multipleOf) else { return false }
            eMin = Swift.max(eMin ?? threshold, threshold)
        }
        if let eMin, let eMax, eMin > eMax { return false }
        return reachability.intersects(lower: eMin, upper: eMax)
    }

    /// Smallest `E` such that `a0 · 10^E` is a multiple of `multiple`, if any.
    private func multipleExponentThreshold(a0: DecimalLiteral, multiple: DecimalLiteral) -> Int? {
        // value = A · 10^(ea0 + E); multiple = M · 10^em.
        // M | A·10^(ea0 + E - em)  ⟺  M | A·10^γ for γ = ea0 + E - em.
        // Valid γ form the interval [G, ∞); find its minimum.
        let a = a0.magnitude
        guard let modulus = DecimalLiteral.uint64Value(multiple.digits) else { return nil }
        var gamma: Int? = nil
        var g = -Swift.min(Self.trailingZeroCount(a.digits), 512)
        while g <= 512 {
            if Self.gammaCondition(a: a, gamma: g, modulus: modulus) {
                gamma = g
                break
            }
            g += 1
        }
        guard let gamma else { return nil }
        return gamma + multiple.exponent - a.exponent
    }

    /// `M | A·10^γ`.
    private static func gammaCondition(a: DecimalLiteral, gamma: Int, modulus: UInt64) -> Bool {
        if gamma >= 0 {
            let remainder = DecimalLiteral.modStreaming(a.digits, modulus: modulus)
            if remainder == 0 { return true }
            return DecimalLiteral.mulMod(remainder, DecimalLiteral.powMod(10, gamma, modulus), modulus) == 0
        }
        let shift = -gamma
        guard a.digits.count >= shift else { return false }
        for offset in 0..<shift where a.digits[a.digits.count - 1 - offset] != 0 { return false }
        let stripped = Array(a.digits.prefix(a.digits.count - shift))
        return DecimalLiteral.modStreaming(stripped, modulus: modulus) == 0
    }

    private static func trailingZeroCount(_ digits: [UInt8]) -> Int {
        var count = 0
        for digit in digits.reversed() where digit == 0 { count += 1 }
        return count
    }
}

// MARK: - Exponent prefix reachability

/// The set of integer exponents reachable from an in-progress exponent prefix.
struct ExponentReachability: Equatable {
    enum Kind: Equatable {
        case any
        case nonNegative
        case nonPositive
        /// Non-negative integers whose canonical digits start with `digits`
        /// (leading zeros stripped; an empty prefix means every non-negative
        /// integer).
        case positivePrefix([UInt8])
        /// Negated `positivePrefix`.
        case negativePrefix([UInt8])
    }

    let kind: Kind

    init(prefix: NumberPrefix.ExponentPrefix) {
        switch prefix {
        case .bare:
            kind = .any
        case .sign(negative: false):
            kind = .nonNegative
        case .sign(negative: true):
            kind = .nonPositive
        case .digits(negative: false, digits: let digits):
            kind = .positivePrefix(Self.strippingLeadingZeros(digits))
        case .digits(negative: true, digits: let digits):
            kind = .negativePrefix(Self.strippingLeadingZeros(digits))
        }
    }

    private static func strippingLeadingZeros(_ digits: [UInt8]) -> [UInt8] {
        var digits = digits
        var first = 0
        while first < digits.count && digits[first] == 0 { first += 1 }
        return Array(digits.dropFirst(first))
    }

    /// Membership for one exponent (used by enum matching).
    func contains(_ exponent: Int) -> Bool {
        switch kind {
        case .any:
            return true
        case .nonNegative:
            return exponent >= 0
        case .nonPositive:
            return exponent <= 0
        case .positivePrefix(let digits):
            guard exponent >= 0 else { return false }
            if digits.isEmpty { return true }
            return Self.digitsMatch(exponent, prefix: digits)
        case .negativePrefix(let digits):
            guard exponent <= 0 else { return false }
            if digits.isEmpty { return true }
            return Self.digitsMatch(-exponent, prefix: digits)
        }
    }

    /// Intersection with an exponent range (`nil` = unbounded on that side).
    func intersects(lower: Int?, upper: Int?) -> Bool {
        if let lower, let upper, lower > upper { return false }
        switch kind {
        case .any:
            return true
        case .nonNegative:
            return upper == nil || upper! >= 0
        case .nonPositive:
            return lower == nil || lower! <= 0
        case .positivePrefix(let digits):
            if digits.isEmpty { return upper == nil || upper! >= 0 }
            return prefixIntersects(digits: digits, negative: false, lower: lower, upper: upper)
        case .negativePrefix(let digits):
            if digits.isEmpty { return lower == nil || lower! <= 0 }
            return prefixIntersects(digits: digits, negative: true, lower: lower, upper: upper)
        }
    }

    /// True when some integer in the range has canonical digits starting with
    /// `digits` (sign flips the window direction).
    private func prefixIntersects(digits: [UInt8], negative: Bool, lower: Int?, upper: Int?) -> Bool {
        let base = DecimalLiteral(negative: false, digits: digits, exponent: 0)
        let lowerValue = lower.map { DecimalLiteral(integer: $0) }
        let upperValue = upper.map { DecimalLiteral(integer: $0) }
        var scale = 0
        while scale < 40 {
            let low = base.scaled(byPowerOf10: scale)
            let high = DecimalLiteral.addMagnitudes(base, .one).scaled(byPowerOf10: scale)
            let windowLow = negative ? DecimalLiteral.addingInteger(high.negated, 1) : low
            let windowHigh = negative ? low.negated : DecimalLiteral.addingInteger(high, -1)
            let entirelyAbove = upperValue.map { windowLow > $0 } ?? false
            let entirelyBelow = lowerValue.map { windowHigh < $0 } ?? false
            if entirelyAbove {
                if !negative { return false }
            } else if entirelyBelow {
                if negative { return false }
            } else {
                return true
            }
            scale += 1
        }
        return false
    }

    private static func digitsMatch(_ value: Int, prefix: [UInt8]) -> Bool {
        let canonical = Array(String(value).utf8).map { $0 - 0x30 }
        guard canonical.count >= prefix.count else { return false }
        return Array(canonical.prefix(prefix.count)) == prefix
    }
}
