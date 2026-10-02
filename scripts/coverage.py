"""Per-entity coverage: real-frame instances (main boxes of every manifest) + approved slices; bar = 40."""
import json, glob, collections
BAR = 40
cls = json.load(open("config/classes_v2.json")); fr = collections.Counter()
seen = set()  # a frame can appear in two manifests (e.g. a batch auto-label and an approved candidate)
for f in glob.glob("data/external/real_v1/manifest_*.json"):
    for x in json.load(open(f)):
        if x["name"] in seen:
            continue
        seen.add(x["name"])
        for l in open(f"data/external/real_v1/labels/{x['name']}.txt").read().splitlines()[:x.get("n_main", 1)]:
            fr[cls[int(l.split()[0])]] += 1
sl = collections.Counter(m["class"] for m in json.load(open("data/slices/approved/manifest.json")))
import yaml
n_base = len(yaml.safe_load(open("data/external/synth_v1/data.yaml"))["names"])  # KataCR's classes have real data
ents = sorted(set(fr) | set(sl) | set(cls[n_base:]), key=lambda e: -(fr[e] + sl[e]))  # new classes count even at 0
print(sum(fr.values()), "labelled instances; pass:", sum(fr[e] + sl[e] >= BAR for e in ents), "of", len(ents))
for e in ents:
    print(f"  {e:28} {fr[e]:4} + {sl[e]:3} = {fr[e]+sl[e]:4} {'OK' if fr[e]+sl[e] >= BAR else 'need ' + str(BAR - fr[e] - sl[e])}")
