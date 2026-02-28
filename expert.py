#!/usr/bin/env python3
"""Collect expert trajectories for MiniGrid EmptyEnv using BFS planning."""

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path

import gymnasium as gym
import minigrid
from minigrid.core.world_object import Goal
from PIL import Image
from tqdm import tqdm

# MiniGrid direction ids: 0=right, 1=down, 2=left, 3=up
DIR_TO_VEC = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}

ACTION_ID_TO_NAME = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

DIR_ID_TO_NAME = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
}

ENVS = [
    "MiniGrid-Empty-Random-6x6-v0",
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-Random-6x6-v0")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("expert_data"))
    parser.add_argument(
        "--instruction",
        type=str,
        default="Go to the green goal square.",
    )
    parser.add_argument(
        "--view-mode",
        choices=["agent", "full"],
        default="agent",
        help="Save only the agent's visible POV ('agent') or the full scene ('full').",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=32,
        help="Tile size for rendered frames.",
    )
    parser.add_argument(
        "--nanovlm-compatible",
        action="store_true",
        help="Also store NanoVLM-friendly fields (`images`, `texts`) in each JSONL row.",
    )
    return parser.parse_args()


def find_goal_pos(env) -> tuple[int, int]:
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if isinstance(obj, Goal):
                return x, y
    raise RuntimeError("Goal cell not found in the environment grid.")


def is_passable(env, x: int, y: int) -> bool:
    grid = env.unwrapped.grid
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return False
    obj = grid.get(x, y)
    return obj is None or isinstance(obj, Goal)


def bfs_actions_to_goal(
    env,
    start_pos: tuple[int, int],
    start_dir: int,
    rng: random.Random,
) -> list[int]:
    goal_pos = find_goal_pos(env)
    start_state = (start_pos[0], start_pos[1], start_dir)
    q = deque([(start_state, [])])

    while q:

        for _ in range(len(q)):
            (x, y, d), path = q.popleft()

            if (x, y) == goal_pos:
                return path

            neighbors = [
                ((x, y, (d - 1) % 4), 0),  # left
                ((x, y, (d + 1) % 4), 1),  # right
            ]
            dx, dy = DIR_TO_VEC[d]
            nx, ny = x + dx, y + dy
            if is_passable(env, nx, ny):
                neighbors.append(((nx, ny, d), 2))  # forward

            for next_state, action in neighbors:
                q.append((next_state, path + [action]))

    raise RuntimeError("No BFS path to goal found.")


def save_rgb_frame(rgb, path: Path) -> None:
    Image.fromarray(rgb).save(path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    dataset_path = args.out_dir / "dataset.jsonl"
    frame_idx = 0
    rows_written = 0

    with (
        dataset_path.open("w", encoding="utf-8") as f,
    ):
        for ep in tqdm(range(args.episodes)):
            env_name = rng.choice(ENVS)
            env = gym.make(env_name, render_mode="rgb_array")
            env.reset(seed=args.seed + ep)
            uenv = env.unwrapped
            
            start_pos = tuple(int(v) for v in uenv.agent_pos)
            start_dir = int(uenv.agent_dir)

            plan = bfs_actions_to_goal(env, start_pos, start_dir, rng)

            for step, action_id in enumerate(plan):
                cur_pos = tuple(int(v) for v in uenv.agent_pos)
                cur_dir = int(uenv.agent_dir)
                if args.view_mode == "agent":
                    rgb = uenv.get_pov_render(tile_size=args.tile_size)
                else:
                    rgb = env.render()
                image_name = f"frame_{frame_idx:08d}.png"
                save_rgb_frame(rgb, args.out_dir / image_name)

                row = {
                    "env_name": env_name,
                    "episode": ep,
                    "step": step,
                    "image": image_name,
                    "action_id": action_id,
                    "agent_pos": [cur_pos[0], cur_pos[1]],
                    "agent_dir": cur_dir,
                }
                if args.nanovlm_compatible:
                    row["images"] = [image_name]
                    row["texts"] = [{
                        "user": (
                            f"{args.instruction}\n"
                            "Predict the next action for the agent. "
                            "Answer with exactly one token: left, right, or forward."
                        ),
                        "assistant": ACTION_ID_TO_NAME[action_id],
                    }]
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                rows_written += 1

                _, _, terminated, truncated, _ = env.step(action_id)
                frame_idx += 1
                if terminated:
                    break
            else:
                raise Exception("Trace wasnt optimmal")

            env.close()


    print(
        f"Saved {rows_written} transitions to {dataset_path}. "
    )


if __name__ == "__main__":

    main()
