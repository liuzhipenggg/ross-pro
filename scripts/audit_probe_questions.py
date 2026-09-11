#!/usr/bin/env python3
"""Apply verified fixes to PROBE_EDIT.json after image audit."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "outputs/recon_pairs_random/PROBE_EDIT.json"
doc = json.load(open(path))

FIXES = {
    "00c": {"answer": "3"},
    "03f": {"question": "Is the license plate text readable on a left parked car?", "answer": "no"},
    "06e": {"answer": "a small pink and white flower"},
    "08g": {"question": "Is there a person standing near the yellow work truck on the right?", "answer": "no"},
    "10a": {"answer": "red / maroon"},
    "20f": {"answer": "15"},
}

REMOVE_IDS = {"07h", "18h", "21g", "21h", "21i", "23h", "23i"}

for it in doc["items"]:
    qs = []
    for q in it.get("questions") or []:
        if q["id"] in REMOVE_IDS:
            continue
        if q["id"] in FIXES:
            q = {**q, **FIXES[q["id"]]}
        qs.append(q)
    it["questions"] = qs

doc["_说明"] = [
    "每张图 questions：针对原图（GT）的问题与标准答案。",
    "2026-08-22 已对照原图人工审核，见 PROBE_AUDIT.md。",
    "跑评测：python scripts/run_probe_vqa_multi.py",
]

json.dump(doc, open(path, "w"), ensure_ascii=False, indent=2)
n = sum(len(it["questions"]) for it in doc["items"])
print(f"fixed PROBE_EDIT.json, {n} questions")
