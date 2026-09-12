#!/usr/bin/env python3
"""Publish the aligned safetensors repack to HuggingFace.

The repack is worth publishing because the alignment is invisible from the
outside: a user who pulls the published checkpoint gets a server that is 4.7 GB
fatter and 3.2x slower on long-context prefill, with no symptom to search for,
while running settings that were measured on the aligned copy.

Requires a token with WRITE scope on the target namespace. A read-only
fine-grained token fails with:
    403 Forbidden: You don't have the rights to create a model under the
    namespace "<user>"

usage: publish_aligned_repack.py <model-dir> <repo-id> [--card PATH]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi, create_repo


def verify(model_dir):
    """Refuse to publish weights whose payload does not match the manifest."""
    manifest = os.path.join(model_dir, "MEI_ALIGN_MANIFEST.json")
    man = json.load(open(manifest))
    for s in man["shards"]:
        path = os.path.join(model_dir, s["file"])
        with open(path, "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            start = 8 + hlen
            if start % man["target_alignment"]:
                sys.exit(f"{s['file']}: data segment at {start} is not aligned")
            f.seek(start)
            h, n = hashlib.sha256(), 0
            while chunk := f.read(1 << 24):
                h.update(chunk)
                n += len(chunk)
        if h.hexdigest() != s["payload_sha256"] or n != s["payload_bytes"]:
            sys.exit(f"{s['file']}: payload does not match the manifest")
        print(f"  {s['file']}: payload verified, segment aligned at {start}")
    print("payload bit-identical to the manifest, every shard aligned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("repo_id")
    ap.add_argument("--card", default=os.path.join(
        os.path.dirname(__file__), "ornith_aligned_model_card.md"))
    args = ap.parse_args()

    verify(args.model_dir)
    api = HfApi()
    print("repo:", create_repo(args.repo_id, repo_type="model", exist_ok=True))

    with tempfile.TemporaryDirectory() as staged:
        for name in os.listdir(args.model_dir):
            # our card replaces the upstream one; upstream assets stay upstream
            if name in ("README.md", "assets"):
                continue
            src = os.path.join(args.model_dir, name)
            if os.path.isfile(src):
                os.symlink(src, os.path.join(staged, name))
        shutil.copy(args.card, os.path.join(staged, "README.md"))
        print("uploading:", sorted(os.listdir(staged)))
        api.upload_folder(
            folder_path=staged, repo_id=args.repo_id, repo_type="model",
            commit_message="Align safetensors data segments; payload bit-identical")
    print("UPLOAD DONE")


if __name__ == "__main__":
    main()
