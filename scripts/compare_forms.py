"""Before/after of the hand-reader fixes: plays per (card, form) for re-read videos, events table version BEFORE
(saved just before re-read round 1) vs now.  usage: uv run python scripts/compare_forms.py [before_version=317]"""
import collections, re, sys
import lancedb

BEFORE = int(sys.argv[1]) if len(sys.argv) > 1 else 317
done = [l.split()[1].rstrip(":") for l in open("data/logs/run_all.log") if l.strip().endswith("reprocessed hand")
        and l[:8] >= "17:2"]  # this round's re-reads only
db = lancedb.connect("data/lancedb")
cur = db.open_table("events")
old = db.open_table("events"); old.checkout(BEFORE)
cols = ["video_id", "kind", "card", "form", "confidence"]
for v in dict.fromkeys(done):
    a = old.search().where(f"video_id = '{v}' and kind = 'play'").limit(10**5).select(cols).to_arrow().to_pylist()
    b = cur.search().where(f"video_id = '{v}' and kind = 'play'").limit(10**5).select(cols).to_arrow().to_pylist()
    ca = collections.Counter((e["card"], e["form"]) for e in a if e["form"] in ("evo", "hero"))
    cb = collections.Counter((e["card"], e["form"]) for e in b if e["form"] in ("evo", "hero"))
    hi = lambda es: sum(e["confidence"] == "high" for e in es)
    print(f"\n{v}: plays {len(a)} -> {len(b)}, high-confidence {hi(a)} -> {hi(b)}")
    for k in sorted(set(ca) | set(cb), key=str):
        if ca[k] != cb[k]:
            print(f"   {k[1]:4} {k[0]:18} {ca[k]:3} -> {cb[k]:3}")
