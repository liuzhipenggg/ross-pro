#!/usr/bin/env python3
"""Merge VLMEval {rank}{world}_{dataset}.pkl shards and run dataset.evaluate."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--judge", default="exact_matching")
    args = p.parse_args()

    ross = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(ross), str(ross / "VLMEvalKit")]
    os.environ.setdefault("LMUData", str(ross / "data" / "LMUData"))

    from vlmeval.dataset import build_dataset
    from vlmeval.smp import dump, load

    run_dir = Path(args.run_dir)
    meta_dir = run_dir.parent
    model = args.model

    for name in args.datasets:
        data_all = {}
        missing = []
        for r in range(args.world_size):
            f = run_dir / f"{r}{args.world_size}_{name}.pkl"
            if not f.exists():
                missing.append(str(f))
                continue
            data_all.update(load(str(f)))
        if missing:
            raise SystemExit(f"{name}: missing shards {missing}")

        dataset = build_dataset(name)
        idx = list(dataset.data["index"])
        if idx and idx[0] not in data_all:
            # pkl keys may be int while dataset index is str (or vice versa)
            data_all = {type(idx[0])(k): v for k, v in data_all.items()}
        miss_idx = [x for x in idx if x not in data_all]
        if miss_idx:
            raise SystemExit(f"{name}: {len(miss_idx)}/{len(idx)} indices missing in pkls")

        data = dataset.data.copy()
        data["prediction"] = [str(data_all[x]) for x in idx]
        if "image" in data:
            data = data.drop(columns=["image"])

        result_file = run_dir / f"{model}_{name}.xlsx"
        dump(data, str(result_file))
        print(f"=> wrote {result_file} n={len(data)}", flush=True)

        ev = dataset.evaluate(str(result_file), model=args.judge, nproc=4, verbose=False)
        print(f"=> evaluated {name}: {ev}", flush=True)

        prefix = f"{model}_{name}"
        for f in run_dir.iterdir():
            if f.name.startswith(prefix) or f.name == "status.json":
                link = meta_dir / f.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(f.resolve())
                print(f"=> link {link} -> {f}", flush=True)


if __name__ == "__main__":
    main()
