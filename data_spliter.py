
#!/usr/bin/env python3
"""Split expert dataset into train/test with strict FOR_TEST episode filtering."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


FOR_TEST = [
    [1, 2],
    [1, 3],
    [2, 3],
]

FOR_TEST_SET = {tuple(pos) for pos in FOR_TEST}


@dataclass
class EpisodeTrace:
    key: tuple[str, int]
    order: int
    rows: list[dict] = field(default_factory=list)
    hits_positions: set[tuple[int, int]] = field(default_factory=set)
    hits_states: set[tuple[int, int, int]] = field(default_factory=set)

    def add_row(self, row: dict) -> None:
        self.rows.append(row)
        agent_pos = row.get("agent_pos")
        if not isinstance(agent_pos, list) or len(agent_pos) != 2:
            return
        pos = (int(agent_pos[0]), int(agent_pos[1]))
        if pos not in FOR_TEST_SET:
            return
        direction = int(row["agent_dir"])
        self.hits_positions.add(pos)
        self.hits_states.add((pos[0], pos[1], direction))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("expert_data/dataset.jsonl"),
        help="Source JSONL with per-step transitions.",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("expert_data/train.jsonl"),
        help="Output JSONL for train split.",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=Path("expert_data/test.jsonl"),
        help="Output JSONL for test split.",
    )
    parser.add_argument(
        "--dropped-out",
        type=Path,
        default=Path("expert_data/dropped_for_test_conflicts.jsonl"),
        help="Output JSONL with dropped FOR_TEST episodes that are kept out of both train/test.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if at least one FOR_TEST position is absent in selected test trajectories.",
    )
    return parser.parse_args()


def _episode_key(row: dict) -> tuple[str, int]:
    env_name = str(row.get("env_name", ""))
    episode = int(row["episode"])
    return env_name, episode


def load_episodes(dataset_path: Path) -> list[EpisodeTrace]:
    episodes: dict[tuple[str, int], EpisodeTrace] = {}
    order: list[tuple[str, int]] = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _episode_key(row)
            if key not in episodes:
                episodes[key] = EpisodeTrace(key=key, order=len(order))
                order.append(key)
            episodes[key].add_row(row)

    return [episodes[key] for key in order]


def select_test_episode_keys(
    episodes: list[EpisodeTrace],
) -> tuple[set[tuple[str, int]], dict[tuple[int, int], tuple[str, int]]]:
    # 1) collect first encountered trajectory for each (position, direction).
    first_by_state: dict[tuple[int, int, int], tuple[str, int]] = {}
    for episode in episodes:
        for state in sorted(episode.hits_states):
            if state not in first_by_state:
                first_by_state[state] = episode.key

    # 2) per FOR_TEST position, keep only one trajectory even if directions differ.
    order_index = {ep.key: ep.order for ep in episodes}
    candidates_by_pos: dict[tuple[int, int], list[tuple[int, int, tuple[str, int]]]] = defaultdict(list)
    for (x, y, direction), key in first_by_state.items():
        candidates_by_pos[(x, y)].append((order_index[key], direction, key))

    episodes_by_key = {ep.key: ep for ep in episodes}
    selected_by_pos: dict[tuple[int, int], tuple[str, int]] = {}
    selected_episode_keys: set[tuple[str, int]] = set()
    covered_positions: set[tuple[int, int]] = set()

    for x, y in FOR_TEST:
        pos = (x, y)
        if pos in covered_positions:
            continue
        candidates = candidates_by_pos.get(pos, [])
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen_key: tuple[str, int] | None = None
        for _, _, candidate_key in candidates:
            candidate_positions = episodes_by_key[candidate_key].hits_positions
            if candidate_positions & covered_positions:
                continue
            chosen_key = candidate_key
            break
        if chosen_key is None:
            # No conflict-free option left for this position.
            continue
        selected_by_pos[pos] = chosen_key
        selected_episode_keys.add(chosen_key)
        covered_positions.update(episodes_by_key[chosen_key].hits_positions)

    return selected_episode_keys, selected_by_pos


def write_split(path: Path, episodes: list[EpisodeTrace]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with path.open("w", encoding="utf-8") as f:
        for episode in episodes:
            for row in episode.rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                rows_written += 1
    return rows_written


def validate_splits(train_episodes: list[EpisodeTrace], test_episodes: list[EpisodeTrace]) -> None:
    for episode in train_episodes:
        if episode.hits_positions:
            raise ValueError(
                f"Train episode {episode.key} touches FOR_TEST positions: {sorted(episode.hits_positions)}"
            )

    test_pos_to_episode: dict[tuple[int, int], set[tuple[str, int]]] = defaultdict(set)
    for episode in test_episodes:
        for pos in episode.hits_positions:
            if pos in FOR_TEST_SET:
                test_pos_to_episode[pos].add(episode.key)

    for pos, owners in test_pos_to_episode.items():
        if len(owners) > 1:
            raise ValueError(f"FOR_TEST position {pos} appears in multiple test trajectories: {sorted(owners)}")


def main() -> None:
    args = parse_args()
    episodes = load_episodes(args.input)

    selected_test_keys, _ = select_test_episode_keys(episodes)
    episodes_with_for_test = [ep for ep in episodes if ep.hits_positions]

    test_episodes = [ep for ep in episodes if ep.key in selected_test_keys]
    train_episodes = [ep for ep in episodes if not ep.hits_positions]
    dropped_episodes = [
        ep
        for ep in episodes_with_for_test
        if ep.key not in selected_test_keys
    ]

    validate_splits(train_episodes, test_episodes)

    covered_by_pos: dict[tuple[int, int], tuple[str, int]] = {}
    for episode in test_episodes:
        for pos in episode.hits_positions:
            if pos in FOR_TEST_SET:
                covered_by_pos[pos] = episode.key

    missing_positions = [tuple(pos) for pos in FOR_TEST if tuple(pos) not in covered_by_pos]
    if missing_positions and args.strict:
        raise ValueError(
            "Missing FOR_TEST positions in selected test split: "
            f"{missing_positions}. Increase dataset coverage or disable --strict."
        )

    train_rows = write_split(args.train_out, train_episodes)
    test_rows = write_split(args.test_out, test_episodes)
    dropped_rows = write_split(args.dropped_out, dropped_episodes)

    print(f"Input episodes: {len(episodes)}")
    print(f"Episodes touching FOR_TEST: {len(episodes_with_for_test)}")
    print(f"Train episodes: {len(train_episodes)} (rows: {train_rows})")
    print(f"Test episodes: {len(test_episodes)} (rows: {test_rows})")
    print(f"Dropped FOR_TEST episodes: {len(dropped_episodes)} (rows: {dropped_rows})")
    print("Selected test trajectories:")
    for pos in FOR_TEST:
        p = tuple(pos)
        chosen = covered_by_pos.get(p)
        if chosen is None:
            print(f"  pos={p}: not found in dataset")
        else:
            print(f"  pos={p}: episode={chosen}")


if __name__ == "__main__":
    main()
