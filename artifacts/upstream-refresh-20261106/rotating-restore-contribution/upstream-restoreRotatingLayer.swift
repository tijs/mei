private func restoreRotatingLayer(
    _ comp: TQDiskSerializer.RotatingLayerComponents,
    into layer: any KVCache
) {
    func apply(_ rot: RotatingKVCache) {
        rot.state = [comp.keys, comp.values]
        rot.metaState = [
            String(comp.keep),
            String(comp.maxSize),
            String(comp.step),
            String(comp.offset),
            String(comp.idx),
        ]
    }
    if let rot = layer as? RotatingKVCache {
        apply(rot)
        return
    }
    // Composite cache that wraps a RotatingKVCache (e.g. DeepseekV4Cache).
    // Restores the inner rotating state; the wrapper's ephemeral buffer
    // state is cleared and will be repopulated on the next prefill.
    if let wrapper = layer as? RotatingKVCacheWrapper {
        apply(wrapper.rotating)
        return
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let rot = cacheList[i] as? RotatingKVCache {
                apply(rot)
                return
            }
            if let wrapper = cacheList[i] as? RotatingKVCacheWrapper {
                apply(wrapper.rotating)
                return
            }
        }
    }
}
