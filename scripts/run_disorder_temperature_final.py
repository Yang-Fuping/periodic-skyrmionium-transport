"""Extend selected paired disorder-temperature strengths to 100 samples."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from _bootstrap import ROOT
from run_disorder_temperature_pilot import identifier, run_pair


STRENGTHS = (0.25, 0.5)
SAMPLES = 100


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    out = ROOT / "results" / "disorder_temperature_joint_v1"
    raw = out / "final_extension.jsonl"
    existing = read_jsonl(raw)
    completed = {row["id"] for row in existing}
    # Samples 0..7 are already available in pilot.jsonl and are combined by the
    # final analyzer.  Only calculate the independent extension here.
    jobs = [(strength, sample) for strength in STRENGTHS for sample in range(8, SAMPLES)
            if identifier(strength, sample) not in completed]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(run_pair, *job): job for job in jobs}
        for position, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with raw.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({
                "completed_new": position, "new_total": len(jobs),
                "stored_extension": len(completed) + position,
                "Wd": record["case"]["Wd"], "sample": record["case"]["sample"],
                "G8_kBT0.01": record["result"]["8"]["thermal"]["0.01"],
            }), flush=True)


if __name__ == "__main__":
    main()
