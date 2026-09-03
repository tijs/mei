import Foundation

/// Schema-aware normalization of tool-call arguments for the OpenAI-compatible
/// API.
///
/// Root cause it fixes: Mei's `GemmaFunctionParser` (vmlx-swift) preserves the
/// model's literal argument spelling into the parsed `arguments` dictionary and
/// `rawArgumentsJSON`. Gemma-4 emits integer-like argument values as JSON
/// *strings* (`{"a":"15","b":"27"}`) while the matching GGUF/reference path
/// emits JSON *numbers* (`{"a":15,"b":27}`). The API contract should report a
/// `number`/`integer` schema field as a JSON number regardless of the model's
/// quoting habits.
///
/// The normalizer walks the *request's* tool definitions (JSON Schema) and:
///   - coerces a numeric-string value to a JSON number where the schema types
///     the field `number` or `integer`;
///   - leaves `string` and `boolean` fields untouched, even numeric-looking
///     strings;
///   - recurses into `object`/`array` fields whose schema declares nested
///     `properties`, coercing only the fields the nested schema types numeric.
///
/// A field with no schema entry, and a property with no `type`, is never
/// guessed at: it passes through verbatim. So legitimate strings are never
/// corrupted, and only schema-typed numeric fields are aligned across model
/// paths.
public enum ToolArgumentNormalizer {
  /// Normalize parsed tool-call arguments against a tool's `parameters`
  /// schema object (the `tools[].function.parameters` from the request).
  public static func normalize(
    arguments: MeiJSONValue,
    parametersSchema: MeiJSONValue?
  ) -> MeiJSONValue {
    guard case .object(let args) = arguments,
      case .object(let parameters)? = parametersSchema,
      case .object(let properties)? = parameters["properties"]
    else {
      return arguments
    }
    return .object(normalizeObject(args, properties: properties))
  }

  private static func normalizeObject(
    _ args: [String: MeiJSONValue],
    properties: [String: MeiJSONValue]
  ) -> [String: MeiJSONValue] {
    var result: [String: MeiJSONValue] = [:]
    result.reserveCapacity(args.count)
    for (key, value) in args {
      guard let property = properties[key], case .object(let prop) = property else {
        result[key] = value
        continue
      }
      result[key] = normalizeValue(value, propertySchema: prop)
    }
    return result
  }

  private static func normalizeValue(
    _ value: MeiJSONValue,
    propertySchema: [String: MeiJSONValue]
  ) -> MeiJSONValue {
    let types = typeNames(propertySchema)
    if types.contains("number") || types.contains("integer") {
      // Schema-typed numeric: coerce a numeric string to a JSON number.
      // Any non-numeric string (or already-numeric value) stays as-is.
      if case .string(let string) = value, let number = parseNumeric(string) {
        return .number(number)
      }
      return value
    }
    if types.contains("object") {
      if case .object(let sub) = value,
        case .object(let subProperties)? = propertySchema["properties"]
      {
        return .object(normalizeObject(sub, properties: subProperties))
      }
      return value
    }
    if types.contains("array") {
      if case .array(let elements) = value,
        case .object(let items)? = propertySchema["items"],
        case .object(let itemProperties)? = items["properties"]
      {
        return .array(
          elements.map { element in
            if case .object(let elementObject) = element {
              return .object(normalizeObject(elementObject, properties: itemProperties))
            }
            return element
          })
      }
      return value
    }
    return value
  }

  /// The declared `type` name(s) of a JSON Schema property, handling both the
  /// single-string form (`"type": "number"`) and the union array form
  /// (`"type": ["number", "null"]`). Empty when untyped.
  private static func typeNames(_ property: [String: MeiJSONValue]) -> [String] {
    switch property["type"] {
    case .string(let name)?:
      return [name]
    case .array(let names)?:
      return names.compactMap { name -> String? in
        if case .string(let string) = name { return string }
        return nil
      }
    default:
      return []
    }
  }

  /// Parse a string as a JSON number (int or float), returning nil when the
  /// string is not fully numeric. Only whole numeric strings qualify — an
  /// empty string, whitespace, or mixed text stays a string.
  static func parseNumeric(_ string: String) -> Double? {
    guard !string.isEmpty else { return nil }
    guard string == string.trimmingCharacters(in: .whitespacesAndNewlines) else { return nil }
    guard let number = Double(string) else { return nil }
    return number
  }
}
