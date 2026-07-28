"""Content-addressed cache of stage outputs, keyed on the recipe (not pixels)."""
from __future__ import annotations
import hashlib
import json
import os
from stages.io import load_fits, save_fits


def recipe_hash(node, input_hashes) -> str:
    payload = {
        "type": node.type,
        "params": {k: node.params[k] for k in sorted(node.params)},
        "inputs": {k: input_hashes[k] for k in sorted(input_hashes)},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def file_sig(path) -> str:
    """mtime (1s resolution) + size; a same-second same-size rewrite may not
    invalidate the cache, which is acceptable vs. hashing large FITS files."""
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def cache_path(work_dir, node_hash, port) -> str:
    return os.path.join(work_dir, "cache", f"{node_hash}__{port}.fits")


def load_cached(path):
    return load_fits(path)


def store_cached(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_fits(path, img)
