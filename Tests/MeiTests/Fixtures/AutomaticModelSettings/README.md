# Automatic model settings fixtures

These are small public metadata files from the four artifact revisions named
in `ModelTuningRegistry`, copied from the corresponding staged bundles. They
contain no model weights. Their independent byte hashes are the production
recognition signatures in `ModelTuningIdentity`.

The tests verify all four profiles, renamed directories, changed/malformed
metadata, ambiguity, stale revision bindings and explicit operator overrides.
Recognition selects settings; it does not certify weight-payload integrity.
