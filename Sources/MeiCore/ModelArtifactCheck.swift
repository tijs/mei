import Foundation

/// Is the model on disk the artifact a profile's settings were measured on?
///
/// This exists because the gap between "downloaded the model" and "serving it
/// well" is invisible. The official Ornith checkpoint is 97% misaligned: 1,421
/// of 1,757 tensors start at an absolute offset safetensors cannot mmap
/// directly, so MLX memcpy's them into anonymous RAM at load. Serving the
/// repacked copy instead is worth 24.28 -> 19.55 GB post-load and a 3.2x faster
/// fresh prefill at 80k (117.6 s -> 36.7 s), and the profile's numbers were
/// measured on the repacked one.
///
/// Nothing about that is discoverable from the outside. A user who follows the
/// install instructions, pulls from HuggingFace and names a profile gets a
/// server that is fatter and slower with no symptom to search for. So Mei
/// checks, and says so.
public enum ModelArtifactCheck {
    public enum Alignment: Equatable {
        case aligned
        case unaligned(unalignedShards: Int, totalShards: Int)
        case unknown(String)
    }

    /// safetensors layout: 8-byte little-endian header length, then that many
    /// bytes of JSON, then the data segment. `data_offsets` are RELATIVE to the
    /// data segment start, so the whole payload is naturally aligned exactly
    /// when the segment start — 8 + headerLength — is 8-byte aligned.
    public static func alignment(ofModelDirectory dir: String) -> Alignment {
        let fm = FileManager.default
        if fm.fileExists(atPath: (dir as NSString)
            .appendingPathComponent("MEI_ALIGN_MANIFEST.json")) {
            return .aligned   // produced by tools/align_safetensors.py
        }
        guard let entries = try? fm.contentsOfDirectory(atPath: dir) else {
            return .unknown("cannot read \(dir)")
        }
        let shards = entries.filter { $0.hasSuffix(".safetensors") }.sorted()
        guard !shards.isEmpty else { return .unknown("no .safetensors in \(dir)") }

        var unaligned = 0
        for shard in shards {
            let path = (dir as NSString).appendingPathComponent(shard)
            guard let h = FileHandle(forReadingAtPath: path) else {
                return .unknown("cannot open \(shard)")
            }
            defer { try? h.close() }
            guard let head = try? h.read(upToCount: 8), head.count == 8 else {
                return .unknown("short read on \(shard)")
            }
            let headerLength = head.withUnsafeBytes {
                $0.loadUnaligned(as: UInt64.self).littleEndian
            }
            if (8 &+ headerLength) % 8 != 0 { unaligned += 1 }
        }
        return unaligned == 0
            ? .aligned
            : .unaligned(unalignedShards: unaligned, totalShards: shards.count)
    }
}
