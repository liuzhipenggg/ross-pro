#!/usr/bin/env python3
"""Rebuild outputs/vlmeval_all_subscores.{csv,md} from VLMEvalKit/outputs/."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROSS = Path(__file__).resolve().parents[1]
OUT_DIR = ROSS / "outputs"
SRC = ROSS / "VLMEvalKit" / "outputs"

MODELS = {
    "llava": "llava-siglip-qwen2-7b-pt558k-sft737k-ftclip",
    "sd15_nox": "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-sft737k-ftclip-ftsd",
    "sd15_xom": "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd",
    "sd35": "ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd",
}
SHORT = ["llava", "sd15_nox", "sd15_xom", "sd35"]
MD_NAMES = {
    "llava": "LLaVA",
    "sd15_nox": "SD15 nox",
    "sd15_xom": "SD15 xom",
    "sd35": "SD35",
}


def as_score(v, *, already_pct: bool, raw: bool = False):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if raw or already_pct:
        return v
    if 0.0 <= v <= 1.5:
        return v * 100.0
    return v


def parse_file(path: Path, bench: str) -> list[tuple[str, str, float]]:
    """Return (split, subcategory, score)."""
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        return [("Overall", str(k), float(v)) for k, v in d.items()]

    df = pd.read_csv(path)
    already = bench in {"POPE", "HallusionBench"}
    raw = bench == "MME"

    if {"cycle", "type", "accuracy"}.issubset(df.columns):
        rows = []
        for _, r in df.iterrows():
            rows.append((str(r["cycle"]), str(r["type"]), as_score(r["accuracy"], already_pct=False)))
        return rows

    if "subject" in df.columns:
        metrics = [c for c in df.columns if c != "subject"]
        rows = []
        for _, r in df.iterrows():
            subj = str(r["subject"])
            for m in metrics:
                rows.append((subj, m, as_score(r[m], already_pct=False)))
        return rows

    if bench == "MME":
        rows = [("Overall", c, as_score(df[c].iloc[0], already_pct=True, raw=True)) for c in df.columns]
        if "perception" in df.columns and "reasoning" in df.columns:
            rows.append(
                (
                    "Overall",
                    "perception+reasoning",
                    float(df["perception"].iloc[0]) + float(df["reasoning"].iloc[0]),
                )
            )
        return rows

    split_col = "split" if "split" in df.columns else None
    metric_cols = [c for c in df.columns if c != split_col]
    rows = []
    for _, r in df.iterrows():
        spl = str(r[split_col]) if split_col else "none"
        for m in metric_cols:
            rows.append((spl, m, as_score(r[m], already_pct=already, raw=raw)))
    return rows


def collect() -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    # bench -> (split, subcat) -> model_key -> value
    out: dict[str, dict[tuple[str, str], dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for key, name in MODELS.items():
        root = SRC / name
        if not root.is_dir():
            continue
        files = []
        for pat in ("*_acc.csv", "*_score.csv", "*_score.json"):
            files.extend(p for p in root.glob(pat) if p.is_file() and not p.is_symlink() or p.is_symlink())
        # only top-level (follow symlinks that sit in model root)
        files = [p for p in files if p.parent == root]
        seen = set()
        for p in files:
            stem = p.name
            for suf in ("_acc.csv", "_score.csv", "_score.json"):
                if stem.endswith(suf) and stem.startswith(name + "_"):
                    bench = stem[len(name) + 1 : -len(suf)]
                    break
            else:
                continue
            if (bench, p.suffix) in seen:
                continue
            seen.add((bench, p.suffix))
            for spl, sub, val in parse_file(p, bench):
                if val is None:
                    continue
                out[bench][(spl, sub)][key] = val
    return out


def fmt(v, nd=4):
    if v is None:
        return ""
    return f"{v:.{nd}f}"


def fmt_md(v, nd=2):
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def delta(a, b):
    if a is None or b is None:
        return ""
    d = b - a
    return f"{d:+.2f}"


def write_csv(data, path: Path) -> pd.DataFrame:
    rows = []
    for bench in sorted(data):
        for (spl, sub) in sorted(data[bench], key=lambda x: (str(x[0]), str(x[1]))):
            rec = data[bench][(spl, sub)]
            rows.append(
                {
                    "benchmark": bench,
                    "split_or_subject": spl,
                    "subcategory": sub,
                    **{k: rec.get(k) for k in SHORT},
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format="%.4f")
    return df


def write_md(data, path: Path) -> None:
    lines = [
        "# VLMEvalKit 全量子分数汇总",
        "",
        "数据来源：`VLMEvalKit/outputs/` 各模型 `*_acc.csv` / `*_score.csv` / `*_score.json`（模型根目录，含指向 T20* 的 symlink）。",
        "",
        "| 列 | 模型 |",
        "|---|---|",
        f"| LLaVA | `{MODELS['llava']}` |",
        f"| SD15 noxomni | `{MODELS['sd15_nox']}` |",
        f"| SD15 xomni | `{MODELS['sd15_xom']}` |",
        f"| SD35 xomni | `{MODELS['sd35']}` |",
        "",
        "除 **MME**（官方 perception/reasoning 原始分）和 **OCRBench** 计数项外，分数为百分制。`—` 表示该模型未跑此集。Δ 相对 LLaVA。",
        "",
        "POPE 的 Overall 列是 **F1**（官方 score.csv）。Hallusion 的 aAcc/fAcc/qAcc 已是百分制。",
        "",
    ]

    for bench in sorted(data):
        keys = list(data[bench])
        splits = {s for s, _ in keys}
        multi_split = len(splits) > 1 or (len(splits) == 1 and next(iter(splits)) not in {"none", "Overall", "val", "test", "2D", "3D"})
        # MMSci / Hallusion / POPE / HRBench / VisOnlyQA: show split column
        show_split = bench in {
            "POPE",
            "HallusionBench",
            "MMSci_DEV_MCQ",
            "HRBench4K",
            "HRBench8K",
            "VisOnlyQA-VLMEvalKit",
        } or (len(splits) > 1)

        lines.append(f"## {bench}")
        lines.append("")
        if show_split:
            header = "| split | metric | LLaVA | SD15 nox | SD15 xom | SD35 | Δ xom vs LLaVA | Δ 35 vs LLaVA |"
            sep = "|---|---|---:|---:|---:|---:|---:|---:|"
        else:
            header = "| subcategory | LLaVA | SD15 nox | SD15 xom | SD35 | Δ xom vs LLaVA | Δ 35 vs LLaVA |"
            sep = "|---|---:|---:|---:|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)

        def sort_key(item):
            spl, sub = item
            # Overall / Average first within a split
            pri = 0 if sub in {"Overall", "all", "Avg", "Final Score Norm", "perception"} else 1
            return (str(spl), pri, str(sub))

        for spl, sub in sorted(keys, key=sort_key):
            rec = data[bench][(spl, sub)]
            vals = [rec.get(k) for k in SHORT]
            # MME raw: 2 decimals is fine; OCR counts 0 decimals-ish but 2 ok
            nd = 2
            cells = [fmt_md(v, nd) for v in vals]
            dx = delta(vals[0], vals[2])
            d35 = delta(vals[0], vals[3])
            if show_split:
                lines.append(f"| {spl} | {sub} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {dx} | {d35} |")
            else:
                lines.append(f"| {sub} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {dx} | {d35} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    data = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "vlmeval_all_subscores.csv"
    md_path = OUT_DIR / "vlmeval_all_subscores.md"
    df = write_csv(data, csv_path)
    write_md(data, md_path)
    print(f"=> {csv_path}  rows={len(df)}  benches={df.benchmark.nunique()}")
    print(f"=> {md_path}")
    print("benches:", ", ".join(sorted(data)))


if __name__ == "__main__":
    main()
