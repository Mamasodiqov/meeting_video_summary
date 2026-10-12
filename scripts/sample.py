"""Sample segments at given minute marks for a quick content overview."""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "raw_segments.json")
segs = json.load(open(OUT, encoding="utf-8"))

for mark_min in (10, 20, 30, 40, 50, 54):
    m = mark_min * 60
    picked = 0
    print(f"=== ~{mark_min} min ===")
    for s in segs:
        if picked >= 2:
            break
        if s["start"] >= m:
            print(f"  {s['start']:8.1f} [{s['language']}] {s['text'][:130]}")
            picked += 1
    print()
