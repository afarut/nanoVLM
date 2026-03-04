#!/usr/bin/env python3
"""GRPO training for text-plan + action generation in MiniGrid EmptyEnv."""

import argparse
import copy
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage
from tqdm import trange

from data.datasets import VQADataset
from data.processors import get_image_processor, get_image_string, get_tokenizer
from models.vision_language_model import VisionLanguageModel


DEFAULT_PROMPT = """### Информация о текущем положении
- Большую часть ты узнаешь из картинки
- Твоё текущее направление: {agent_dir}
    - 0: right
    - 1: down
    - 2: left
    - 3: up
### Твоя глобальная задача:
- Дойти до зеленого квадратика.
### Твоя локальная задача
- Cделать один шаг в сторону глобальной задачи.

### Формат вывода (строго):
План: 2-3 коротких предложения о том, где агент и куда лучше сделать следующий шаг.
Действие: Число
- Число должно быть от 0 до 2 включительно
- 0: Turn left
- 1: Turn right
- 2: Move forward
- ВАЖНО: после строки "Действие:" должно быть только одно число.

Сначала напиши "План:", потом "Действие:".
"""


ACTION_RE = re.compile(r"действие\s*:\s*([0-2])", re.IGNORECASE)
ANY_ACTION_RE = re.compile(r"(?<!\d)([0-2])(?!\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model", type=str, default="Afarut/nanoVLM-sft")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-Random-6x6-v0")
    parser.add_argument("--episode-size", type=int, default=15)
    parser.add_argument("--test-dataset", type=str, default="Afarut/NanoVLMEmptyEnv")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--train-episodes", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gen-max-new-tokens", type=int, default=48)
    parser.add_argument("--gen-top-k", type=int, default=50)
    parser.add_argument("--gen-top-p", type=float, default=0.9)
    parser.add_argument("--gen-temperature", type=float, default=0.8)
    parser.add_argument("--invalid-action-fallback", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--format-reward", type=float, default=0.10)
    parser.add_argument("--action-parse-reward", type=float, default=0.10)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--shaping-coef", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("grpo_runs"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_goal_pos(env) -> tuple[int, int]:
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is not None and obj.type == "goal":
                return (x, y)
    raise RuntimeError("Goal not found in MiniGrid map.")


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def resolve_action_token_ids(tokenizer) -> list[int]:
    action_ids = []
    for action_str in ("0", "1", "2"):
        candidates: list[int] = []
        for variant in (action_str, f" {action_str}"):
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if len(tok) == 1:
                candidates.append(tok[0])

        if not candidates:
            for tok_id in range(tokenizer.vocab_size):
                decoded = tokenizer.decode([tok_id]).strip()
                if decoded == action_str:
                    candidates.append(tok_id)
                    break

        if not candidates:
            raise RuntimeError(f"Could not find single-token id for action '{action_str}'.")

        action_ids.append(candidates[0])

    if len(set(action_ids)) != 3:
        raise RuntimeError(f"Action token ids are not unique: {action_ids}")
    return action_ids


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_state_tensors(model, tokenizer, image_processor, uenv, tile_size: int, device: torch.device):
    rgb = uenv.get_pov_render(tile_size=tile_size)
    image = PILImage.fromarray(rgb)
    image_tensor, split_ratio = image_processor(image.convert("RGB"))

    if not hasattr(tokenizer, "global_image_token") and split_ratio[0] * split_ratio[1] == len(image_tensor) - 1:
        image_tensor = image_tensor[1:]

    image_string = get_image_string(tokenizer, [split_ratio], model.cfg.mp_image_token_length)
    prompt = DEFAULT_PROMPT.format(agent_dir=int(uenv.agent_dir))
    messages = [{"role": "user", "content": image_string + prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=device)
    images = image_tensor.to(device)
    return input_ids, attention_mask, images


def next_token_logits(model, input_ids, attention_mask, images):
    hidden, _ = model(input_ids=input_ids, images=images, attention_mask=attention_mask)
    logits = hidden[:, -1, :]
    if not model.decoder.lm_use_tokens:
        logits = model.decoder.head(logits)
    return logits.squeeze(0)


def parse_action_from_text(text: str) -> int | None:
    m = ACTION_RE.search(text)
    if m:
        return int(m.group(1))
    m = ANY_ACTION_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def count_sentences(text: str) -> int:
    pieces = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    return len(pieces)


def format_bonus(text: str, action: int | None, format_reward: float, action_parse_reward: float) -> float:
    bonus = 0.0
    if "план" in text.lower() and 2 <= count_sentences(text) <= 5:
        bonus += format_reward
    if action is not None:
        bonus += action_parse_reward
    return bonus


@torch.no_grad()
def _prompt_tensors_from_row(
    test_dataset: VQADataset,
    row: dict,
    device: torch.device,
):
    images_data = row["images"]
    if images_data is None:
        images_data = []
    if not isinstance(images_data, list):
        images_data = [images_data]

    processed_images, splitted_image_counts = test_dataset._process_images(images_data)
    messages = test_dataset._get_messages(row, splitted_image_counts)
    prompt_messages = [m for m in messages if m["role"] == "user"]
    if not prompt_messages:
        raise RuntimeError("No user prompt in test row.")

    encoded = test_dataset.tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=device)
    images = [img.to(device) for img in processed_images]
    return input_ids, attention_mask, images


def load_test_rows(dataset_name: str, split: str) -> list[dict]:
    ds = load_dataset(dataset_name, split=split)
    ds = ds.cast_column("images", HFImage(decode=False))
    rows = []
    for i in range(len(ds)):
        row = ds[i]
        image_cell = row["images"]
        if isinstance(image_cell, dict) and image_cell.get("bytes") is not None:
            row["images"] = PILImage.open(io.BytesIO(image_cell["bytes"])).convert("RGB")
        elif isinstance(image_cell, dict) and image_cell.get("path") is not None:
            row["images"] = PILImage.open(image_cell["path"]).convert("RGB")
        elif isinstance(image_cell, str):
            row["images"] = PILImage.open(image_cell).convert("RGB")
        else:
            raise ValueError("Unsupported image format in test dataset row.")
        rows.append(row)
    return rows


def build_full_attention_mask(base_mask: torch.Tensor, extra_len: int) -> torch.Tensor:
    if extra_len <= 0:
        return base_mask
    extra = torch.ones((base_mask.size(0), extra_len), dtype=base_mask.dtype, device=base_mask.device)
    return torch.cat([base_mask, extra], dim=1)


def sequence_logprob(model, input_ids, attention_mask, images, generated_ids: torch.Tensor) -> torch.Tensor:
    if generated_ids.numel() == 0:
        return torch.zeros((), device=input_ids.device)

    prompt_len = input_ids.size(1)
    gen_len = generated_ids.numel()
    # `generate()` returns inference tensors; clone to use as gather indices in autograd path.
    gen_ids = generated_ids.clone().unsqueeze(0)
    full_ids = torch.cat([input_ids, gen_ids], dim=1)
    full_mask = build_full_attention_mask(attention_mask, gen_len)

    logits, _ = model(input_ids=full_ids, images=images, attention_mask=full_mask)
    if not model.decoder.lm_use_tokens:
        logits = model.decoder.head(logits)

    token_logits = logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
    log_probs = F.log_softmax(token_logits, dim=-1)
    selected = log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
    return selected.sum()


def one_step_shaped_reward(env, action: int, goal_pos: tuple[int, int], shaping_coef: float) -> float:
    sim_env = copy.deepcopy(env)
    sim_uenv = sim_env.unwrapped
    before = manhattan((int(sim_uenv.agent_pos[0]), int(sim_uenv.agent_pos[1])), goal_pos)
    _, env_reward, terminated, truncated, _ = sim_env.step(action)
    after = manhattan((int(sim_uenv.agent_pos[0]), int(sim_uenv.agent_pos[1])), goal_pos)
    sim_env.close()

    shaped = float(env_reward) + shaping_coef * float(before - after)
    if truncated:
        shaped -= 0.1
    if terminated and env_reward > 0:
        shaped += 0.5
    return shaped


@torch.no_grad()
def evaluate_on_test_dataset(
    model,
    test_dataset: VQADataset,
    test_rows: list[dict],
    action_token_ids: list[int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ce_losses = []
    traj_log_probs: dict[tuple, float] = defaultdict(float)

    for i, row in enumerate(test_rows):
        processed = test_dataset[i]
        input_ids_full = processed["input_ids"].unsqueeze(0).to(device)
        attention_full = processed["attention_mask"].unsqueeze(0).to(device)
        labels_full = processed["labels"].unsqueeze(0).to(device)
        images_full = [img.to(device) for img in processed["images"]]
        _, ce_loss = model(
            input_ids=input_ids_full,
            images=images_full,
            attention_mask=attention_full,
            targets=labels_full,
        )
        ce_losses.append(float(ce_loss.item()))

        input_ids, attention_mask, images = _prompt_tensors_from_row(
            test_dataset=test_dataset,
            row=row,
            device=device,
        )
        logits = next_token_logits(model, input_ids, attention_mask, images)
        action_logits = logits[action_token_ids]
        log_probs = F.log_softmax(action_logits, dim=-1)

        action_id = int(row["action_id"])
        traj_key = (row.get("env_name", ""), int(row["episode"]))
        traj_log_probs[traj_key] += float(log_probs[action_id].item())

    if not ce_losses:
        return {"test_ce": 0.0, "best_traj_prob": 0.0, "best_traj_logprob": float("-inf")}

    all_log_probs = list(traj_log_probs.values())
    best_traj_logprob = max(all_log_probs)
    best_traj_prob = float(np.exp(best_traj_logprob))

    return {
        "test_ce": float(np.mean(ce_losses)),
        "best_traj_prob": best_traj_prob,
        "best_traj_logprob": float(best_traj_logprob),
    }


def save_plots(metrics: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(metrics["train_loss"], lw=1.2)
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Episode")

    axes[1].plot(metrics["eval_steps"], metrics["eval_test_ce_grpo"], label="GRPO text+action", lw=1.7)
    axes[1].axhline(metrics["eval_test_ce_sft"], linestyle="--", label="SFT baseline", lw=1.7)
    axes[1].set_title("Test CrossEntropyLoss")
    axes[1].set_xlabel("Episode")
    axes[1].legend()

    axes[2].plot(metrics["eval_steps"], metrics["eval_best_traj_prob_grpo"], label="GRPO text+action", lw=1.7)
    axes[2].axhline(metrics["eval_best_traj_prob_sft"], linestyle="--", label="SFT baseline", lw=1.7)
    axes[2].set_title("Best Trajectory Probability (test)")
    axes[2].set_xlabel("Episode")
    axes[2].set_yscale("log")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    policy = VisionLanguageModel.from_pretrained(args.sft_model).to(device)
    reference = VisionLanguageModel.from_pretrained(args.sft_model).to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    tokenizer = get_tokenizer(policy.cfg.lm_tokenizer, policy.cfg.vlm_extra_tokens, policy.cfg.lm_chat_template)
    image_processor = get_image_processor(
        policy.cfg.max_img_size,
        policy.cfg.vit_img_size,
        getattr(policy.cfg, "resize_to_max_side_len", False),
    )

    action_token_ids = resolve_action_token_ids(tokenizer)
    test_rows = load_test_rows(args.test_dataset, args.test_split)
    test_dataset = VQADataset(
        test_rows,
        tokenizer,
        image_processor,
        policy.cfg.mp_image_token_length,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    print("Evaluating SFT baseline on test dataset...")
    sft_eval = evaluate_on_test_dataset(
        reference,
        test_dataset,
        test_rows,
        action_token_ids,
        device,
    )
    print(
        f"SFT baseline test: CE={sft_eval['test_ce']:.6f}, "
        f"best_traj_prob={sft_eval['best_traj_prob']:.6e}"
    )

    metrics = {
        "train_loss": [],
        "train_success": [],
        "train_return": [],
        "eval_steps": [],
        "eval_test_ce_grpo": [],
        "eval_best_traj_prob_grpo": [],
        "eval_best_traj_logprob_grpo": [],
        "eval_test_ce_sft": sft_eval["test_ce"],
        "eval_best_traj_prob_sft": sft_eval["best_traj_prob"],
        "eval_best_traj_logprob_sft": sft_eval["best_traj_logprob"],
        "sft_eval": sft_eval,
    }

    pbar = trange(args.train_episodes, desc="GRPO text+action train", leave=True)
    for ep in pbar:
        policy.train()
        env = gym.make(args.env_id, render_mode="rgb_array", size=args.episode_size)
        _, _ = env.reset(seed=args.seed + ep)
        goal_pos = find_goal_pos(env)

        ep_return = 0.0
        ep_loss_values = []
        success = 0

        for _ in range(args.max_steps):
            input_ids, attention_mask, images = prepare_state_tensors(
                policy,
                tokenizer,
                image_processor,
                env.unwrapped,
                args.tile_size,
                device,
            )

            sampled_rewards = []
            sampled_log_probs = []
            sampled_ref_log_probs = []
            sampled_actions = []

            for _g in range(args.group_size):
                generated = policy.generate(
                    input_ids,
                    images,
                    attention_mask=attention_mask,
                    max_new_tokens=args.gen_max_new_tokens,
                    top_k=args.gen_top_k,
                    top_p=args.gen_top_p,
                    temperature=args.gen_temperature,
                    greedy=False,
                )
                generated_ids = generated[0]
                output_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
                action = parse_action_from_text(output_text)

                if action is None:
                    action_for_reward = random.randint(0, 2)
                else:
                    action_for_reward = int(action)

                reward = one_step_shaped_reward(env, action_for_reward, goal_pos, args.shaping_coef)
                reward += format_bonus(output_text, action, args.format_reward, args.action_parse_reward)

                seq_log_prob = sequence_logprob(policy, input_ids, attention_mask, images, generated_ids)
                with torch.no_grad():
                    seq_ref_log_prob = sequence_logprob(reference, input_ids, attention_mask, images, generated_ids)

                sampled_rewards.append(reward)
                sampled_log_probs.append(seq_log_prob)
                sampled_ref_log_probs.append(seq_ref_log_prob)
                sampled_actions.append(action)

            rewards_t = torch.tensor(sampled_rewards, dtype=torch.float32, device=device)
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std(unbiased=False) + 1e-6)

            sel_log_probs = torch.stack(sampled_log_probs)
            sel_ref_log_probs = torch.stack(sampled_ref_log_probs).detach()

            loss_pg = -(advantages.detach() * sel_log_probs).mean()
            loss_kl = (sel_log_probs - sel_ref_log_probs).mean()
            loss = loss_pg + args.kl_coef * loss_kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            ep_loss_values.append(float(loss.item()))

            action_exec = sampled_actions[0]
            if action_exec is None:
                action_exec = args.invalid_action_fallback

            _, reward, terminated, truncated, _ = env.step(int(action_exec))
            ep_return += float(reward)
            if terminated and reward > 0:
                success = 1
            if terminated or truncated:
                break

        env.close()
        metrics["train_loss"].append(float(np.mean(ep_loss_values)) if ep_loss_values else 0.0)
        metrics["train_success"].append(success)
        metrics["train_return"].append(ep_return)
        pbar.set_postfix({"success": success, "ret": f"{ep_return:.2f}"})

        if (ep + 1) % args.eval_every == 0:
            eval_stats = evaluate_on_test_dataset(
                policy,
                test_dataset,
                test_rows,
                action_token_ids,
                device,
            )

            metrics["eval_steps"].append(ep + 1)
            metrics["eval_test_ce_grpo"].append(eval_stats["test_ce"])
            metrics["eval_best_traj_prob_grpo"].append(eval_stats["best_traj_prob"])
            metrics["eval_best_traj_logprob_grpo"].append(eval_stats["best_traj_logprob"])

            print(
                f"[Eval @ {ep + 1}] test_CE={eval_stats['test_ce']:.6f}, "
                f"best_traj_prob={eval_stats['best_traj_prob']:.6e}"
            )

    model_dir = args.output_dir / "grpo_text_action_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(model_dir))

    plot_path = args.output_dir / "grpo_text_action_learning_curves.png"
    save_plots(metrics, plot_path)

    metrics_path = args.output_dir / "metrics_text_action.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved GRPO text+action model to: {model_dir}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
