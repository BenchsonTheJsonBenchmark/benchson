"""Creates (offline, read-only on the baseline) and reports the published-baseline-anchored schema pool.

    python src/export_anchor_pool.py --baseline-dir data_published/<sha12>            # write the anchor file + print statistics
    python src/export_anchor_pool.py --baseline-dir data_published/<sha12> --stats    # statistics of an existing anchor file

The anchor file `published_train_schemas.json` holds every distinct schema of the published TRAIN rows (id, content,
source/subset/name, content hash) plus the id/hash/structure of the published TEST schemas. It is written next to
`reserved_schemas.json` (never overwritten) and verified against it on every load. Statistics are computed from the
published data with the explicit size restriction (--max-schema-kb, default 10).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data_generation.anchor import ANCHOR_FILENAME, AnchorError, load_anchor, select_pool, write_anchor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--out", default=None, help=f"anchor file (default <baseline-dir>/{ANCHOR_FILENAME})")
    ap.add_argument("--max-schema-kb", type=float, default=10)
    ap.add_argument("--stats", action="store_true", help="do not write; report on the existing anchor file")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.baseline_dir, ANCHOR_FILENAME)
    reserved = os.path.join(args.baseline_dir, "reserved_schemas.json")
    try:
        if not args.stats:
            write_anchor(args.baseline_dir, out)
            print(f"wrote {out}")
        anchor = load_anchor(out, reserved if os.path.isfile(reserved) else None)
    except AnchorError as e:
        print(f"FAILED: {e}")
        return 1
    kept, stats = select_pool(anchor, args.max_schema_kb)
    sizes = sorted(e["size_bytes"] for e in anchor["train"])
    stats["published_train_schema_size_bytes"] = {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]}
    stats["published_train_by_source"] = {}
    for e in anchor["train"]:
        k = "/".join(p for p in (e["source"], e["subset"]) if p) or "unknown"
        stats["published_train_by_source"][k] = stats["published_train_by_source"].get(k, 0) + 1
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
