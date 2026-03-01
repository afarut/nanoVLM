#!/usr/bin/env python3
"""Remove duplicated train samples by full state key (agent_pos_x, agent_pos_y, agent_dir)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("expert_data/train.jsonl"),
        help="Path to source train JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("expert_data/train_clean.jsonl"),
        help="Path to cleaned train JSONL.",
    )
    return parser.parse_args()


def row_state_key(row: dict) -> tuple[int, int, int]:
    agent_pos = row["agent_pos"]
    if not isinstance(agent_pos, list) or len(agent_pos) != 2:
        raise ValueError("Row has invalid agent_pos, expected list[int, int].")
    return int(agent_pos[0]), int(agent_pos[1]), int(row["agent_dir"])


def clean_train_dataset(input_path: Path, output_path: Path) -> tuple[int, int]:
    seen_states: set[tuple[int, int, int]] = set()
    total_rows = 0
    kept_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            total_rows += 1
            row = json.loads(line)
            state_key = row_state_key(row)
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            dst.write(json.dumps(row, ensure_ascii=True) + "\n")
            kept_rows += 1

    return total_rows, kept_rows


def main() -> None:
    args = parse_args()
    total_rows, kept_rows = clean_train_dataset(args.input, args.output)
    removed_rows = total_rows - kept_rows
    print(f"Input rows: {total_rows}")
    print(f"Kept rows: {kept_rows}")
    print(f"Removed duplicates: {removed_rows}")
    print(f"Saved cleaned train dataset to: {args.output}")


if __name__ == "__main__":
    main()
