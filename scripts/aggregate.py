from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_report(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "model",
                "dataset",
                "metric_name",
                "score",
                "time_sec",
                "mem_mb",
                "checkpoint_dir",
                "source_file",
            ]
        )

    df = pd.DataFrame(records)
    expected_columns = [
        "model",
        "dataset",
        "metric_name",
        "score",
        "time_sec",
        "mem_mb",
        "checkpoint_dir",
        "results_file",
    ]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = ""

    if "source_file" not in df.columns:
        df["source_file"] = df["results_file"]

    ordered_columns = expected_columns + ["source_file"]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    return df[ordered_columns + remaining_columns]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate standalone JSON result files into a CSV report.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results_report.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    records: List[Dict] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            record = load_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            record.setdefault("source_file", path.name)
            records.append(record)

    report = build_report(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Saved {output_path} with shape {report.shape}")


if __name__ == "__main__":
    main()