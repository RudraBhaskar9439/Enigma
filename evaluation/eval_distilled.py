"""
eval_distilled.py — Produce the four hackathon-grade plots that prove
the distilled student learned something on DropoutCommonsEnv.

Outputs:
  docs/img/reward_curve.png      cumulative reward, trained vs baselines
  docs/img/action_fidelity.png   teacher↔student action-vector scatter (R²)
  docs/img/per_intervention.png  per-intervention recommendation comparison
  results.json                   all numbers, for embedding in the README

Usage (after training):
  python evaluation/eval_distilled.py \\
      --adapter vishwamitra-1b-lora/ \\
      --val      data/val.jsonl \\
      --episodes 50

Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.dropout_env import DropoutCommonsEnv  # noqa: E402
from env.scenarios.funding_cut import FundingCutScenario  # noqa: E402

ACTION_NAMES = [
    "funding_boost", "teacher_incentive", "student_scholarship", "attendance_mandate",
    "resource_realloc", "transparency_report", "staff_hiring", "counseling_programs",
]

# Must mirror generate_dataset.py:SYSTEM_PROMPT exactly — the model was
# fine-tuned on these tokens; even small wording drift hurts inference.
SYSTEM_PROMPT = (
    "You are an educational policy assistant. Given a system state and a "
    "scenario brief, recommend intervention intensities (each in [0, 1]) and "
    "provide a brief rationale. Output a single JSON object with keys "
    "`action_vector` (eight named floats) and `reasoning` (1-3 sentences)."
)

# Maps each slot of env.state.SystemState.to_obs_array() to the training-format
# field name (or None to drop). policy_compliance is reused as trust_score
# because the env doesn't expose a separate trust_score field.
_STATE_FIELD_MAP: list[tuple[str | None, Any]] = [
    ("enrollment_rate",     lambda x: round(float(x), 3)),
    ("attendance_rate",     lambda x: round(float(x), 3)),
    ("dropout_rate",        lambda x: round(float(x), 3)),
    ("teacher_retention",   lambda x: round(float(x), 3)),
    (None, None),
    ("avg_class_size",      lambda x: int(round(float(x) * 60))),
    (None, None),
    ("resource_allocation", lambda x: round(float(x), 3)),
    ("student_engagement",  lambda x: round(float(x), 3)),
    ("teacher_burnout",     lambda x: round(float(x), 3)),
    ("trust_score",         lambda x: round(float(x), 3)),
    ("budget_remaining",    lambda x: float(round(float(x) * 2_000_000 / 1000) * 1000)),
    ("step",                lambda x: int(x)),
]


def _obs_to_state_dict(obs) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for i, (name, conv) in enumerate(_STATE_FIELD_MAP):
        if name is None or i >= len(obs):
            continue
        state[name] = conv(obs[i])
    return state


def _format_user_prompt(state: dict[str, Any], scenario: str) -> str:
    state_lines = "\n".join(f"  - {k}: {state[k]}" for k in state)
    return (
        f"STATE:\n{state_lines}\n\n"
        f"SCENARIO: {scenario}\n\n"
        "Respond with a single JSON object in this exact shape:\n"
        '{"action_vector": {"funding_boost": <float>, '
        '"teacher_incentive": <float>, "student_scholarship": <float>, '
        '"attendance_mandate": <float>, "resource_realloc": <float>, '
        '"transparency_report": <float>, "staff_hiring": <float>, '
        '"counseling_programs": <float>}, '
        '"reasoning": "<one to three sentences>"}'
    )


_MODEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}


def _pick_device():
    """Single-device placement, never device_map='auto'.

    'auto' triggers accelerate to plan a (possibly disk-offloaded) layout,
    which on macOS/MPS hits a known peft bug in `_update_offload` where
    `extended_prefix` carries one stray `model.` segment too many. Explicit
    single-device placement avoids that code path entirely.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _load_model_and_tokenizer(adapter_path: str):
    if adapter_path in _MODEL_CACHE:
        return _MODEL_CACHE[adapter_path]

    import traceback
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    base_id = "unsloth/Llama-3.2-1B-Instruct"
    device, dtype = _pick_device()
    print(f"  Loading base model {base_id} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype)
    base = base.to(device)

    print(f"  Loading LoRA adapter from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base, adapter_path)
    except Exception:
        traceback.print_exc()
        print("  ⚠ Adapter load failed on accelerator; retrying on CPU...")
        del base
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.float32,
        ).to("cpu")
        model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    _MODEL_CACHE[adapter_path] = (model, tokenizer, torch)
    return _MODEL_CACHE[adapter_path]


def _generate_action_vector(model, tokenizer, torch, messages: list[dict]) -> np.ndarray:
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
    )
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else None
        av = data["action_vector"] if data and "action_vector" in data else {}
        return np.array(
            [float(av.get(n, 0.0)) for n in ACTION_NAMES],
            dtype=np.float32,
        )
    except Exception:
        return np.full(8, 0.5, dtype=np.float32)

# ============================================================================
# Plot styling — consistent across all four figures
# ============================================================================
COLOR = {
    "trained":      "#3b82f6",   # blue
    "random":       "#f87171",   # red
    "zero":         "#94a3b8",   # slate
    "teacher":      "#fbbf24",   # amber
    "background":   "#0f172a",
    "grid":         "#334155",
    "annotation":   "#10b981",   # emerald accents
    "intervention": [
        "#3b82f6", "#fbbf24", "#10b981", "#a78bfa",
        "#f59e0b", "#ec4899", "#06b6d4", "#84cc16",
    ],
}


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply our editorial style — clean, high-contrast, paper-grade."""
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9)


# ============================================================================
# Policies
# ============================================================================

def random_policy(rng: np.random.Generator):
    """Uniform random in [0, 1]^8."""
    def _f(_obs):
        return rng.uniform(0.0, 1.0, size=8).astype(np.float32)
    return _f


def zero_policy():
    """Do nothing every step."""
    def _f(_obs):
        return np.zeros(8, dtype=np.float32)
    return _f


_ROLLOUT_SCENARIO_BRIEF = (
    "Ongoing DropoutCommonsEnv step in a school district under a funding-cut "
    "scenario: budget pressure, rising teacher burnout, and dropout signals. "
    "Recommend the next round of intervention intensities."
)


def make_trained_policy(adapter_path: str):
    """Load the LoRA adapter and return a callable obs → action."""
    try:
        model, tokenizer, torch = _load_model_and_tokenizer(adapter_path)
    except Exception as e:
        print(f"  ⚠ Could not load trained model: {e}")
        print("  Falling back to zero policy for the trained slot.")
        return zero_policy()

    def _f(obs):
        state = _obs_to_state_dict(obs)
        user_msg = _format_user_prompt(state, _ROLLOUT_SCENARIO_BRIEF)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        return _generate_action_vector(model, tokenizer, torch, messages)
    return _f


# ============================================================================
# Episode roller
# ============================================================================
@dataclass
class EpisodeRollout:
    rewards: list[float]
    cumulative: list[float]
    terminated: bool
    final_step: int


def rollout(env, policy, max_steps: int = 100) -> EpisodeRollout:
    obs, _info = env.reset()
    rewards, cum = [], []
    total = 0.0
    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        rewards.append(float(reward))
        total += float(reward)
        cum.append(total)
        if terminated or truncated:
            return EpisodeRollout(rewards, cum, terminated, step + 1)
    return EpisodeRollout(rewards, cum, False, max_steps)


def evaluate_policy(name: str, policy, n_episodes: int, max_steps: int, seed_base: int = 0):
    """Roll the policy out for n_episodes; collect cumulative reward over time."""
    print(f"  Evaluating {name} for {n_episodes} episodes...")
    all_cum = []
    final_rewards = []
    n_collapsed = 0
    for ep in range(n_episodes):
        env = DropoutCommonsEnv(
            scenario=FundingCutScenario(), episode_length=max_steps,
        )
        env.reset(seed=seed_base + ep)
        ro = rollout(env, policy, max_steps=max_steps)
        # Pad the cumulative trace to max_steps so we can stack across episodes
        cum = list(ro.cumulative)
        if len(cum) < max_steps:
            cum.extend([cum[-1] if cum else 0.0] * (max_steps - len(cum)))
        all_cum.append(cum)
        final_rewards.append(cum[-1])
        if ro.terminated:
            n_collapsed += 1
    arr = np.array(all_cum)
    return {
        "name": name,
        "all_cumulative": arr,                          # (n_episodes, max_steps)
        "mean_cumulative": arr.mean(axis=0),
        "se_cumulative":   arr.std(axis=0) / max(1.0, math.sqrt(arr.shape[0])),
        "final_rewards":   np.array(final_rewards),
        "mean_final":      float(np.mean(final_rewards)),
        "std_final":       float(np.std(final_rewards)),
        "n_collapsed":     int(n_collapsed),
        "episodes_solved": n_episodes - n_collapsed,
    }


# ============================================================================
# Action-vector fidelity (student vs. teacher on val set)
# ============================================================================

def evaluate_fidelity(adapter_path: str, val_path: Path) -> dict[str, Any]:
    print(f"  Computing student↔teacher fidelity on {val_path}...")
    try:
        model, tokenizer, torch = _load_model_and_tokenizer(adapter_path)
    except Exception as e:
        print(f"  ⚠ Could not load adapter: {e}")
        return {"available": False}

    teacher_vectors, student_vectors = [], []
    with val_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"  Generating predictions on {len(rows)} validation states...")
    for i, row in enumerate(rows):
        try:
            target = json.loads(row["messages"][-1]["content"])
            teacher_av = target["action_vector"]
            teacher = np.array(
                [float(teacher_av.get(n, 0)) for n in ACTION_NAMES], dtype=np.float32,
            )
        except Exception:
            continue

        # Replay the EXACT prompt used during training — no reconstruction.
        messages = [m for m in row["messages"] if m["role"] != "assistant"]
        student = _generate_action_vector(model, tokenizer, torch, messages)

        teacher_vectors.append(teacher)
        student_vectors.append(student)

        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(rows)}]")

    if not teacher_vectors:
        return {"available": False}

    teacher_arr = np.array(teacher_vectors)   # (N, 8)
    student_arr = np.array(student_vectors)   # (N, 8)

    # Per-intervention metrics
    mae_per = np.mean(np.abs(teacher_arr - student_arr), axis=0)
    overall_mae = float(np.mean(mae_per))
    # Pearson correlation, flattened
    flat_t = teacher_arr.flatten()
    flat_s = student_arr.flatten()
    if flat_t.std() > 1e-6 and flat_s.std() > 1e-6:
        pearson = float(np.corrcoef(flat_t, flat_s)[0, 1])
    else:
        pearson = 0.0

    # Top-3 agreement: how often is the student's top-3 == teacher's top-3 (set)?
    top3_match = 0
    for t, s in zip(teacher_arr, student_arr):
        t_top = set(np.argsort(-t)[:3].tolist())
        s_top = set(np.argsort(-s)[:3].tolist())
        if t_top == s_top:
            top3_match += 1
    top3_agreement = top3_match / len(teacher_arr) if teacher_arr.size else 0.0

    return {
        "available":       True,
        "n_examples":      int(len(teacher_arr)),
        "mae_per_intervention": {ACTION_NAMES[i]: float(mae_per[i]) for i in range(8)},
        "overall_mae":     overall_mae,
        "pearson":         pearson,
        "top3_agreement":  float(top3_agreement),
        "teacher_arr":     teacher_arr,
        "student_arr":     student_arr,
    }


# ============================================================================
# Plot 1: Reward improvement curve
# ============================================================================

def plot_reward_curve(results: list[dict], outpath: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in results:
        c = COLOR.get(r["name"], "#888")
        x = np.arange(len(r["mean_cumulative"]))
        ax.plot(x, r["mean_cumulative"], label=r["name"].title(), color=c, linewidth=2.2)
        ax.fill_between(
            x,
            r["mean_cumulative"] - r["se_cumulative"],
            r["mean_cumulative"] + r["se_cumulative"],
            color=c, alpha=0.18,
        )

    style_axes(
        ax,
        title="Distilled student vs. baselines on DropoutCommonsEnv",
        xlabel="Step within episode",
        ylabel="Cumulative reward (mean ± SE)",
    )
    ax.legend(loc="best", frameon=False, fontsize=10)
    ax.axhline(0, color=COLOR["grid"], linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved {outpath}")


# ============================================================================
# Plot 2: Action-vector fidelity scatter
# ============================================================================

def plot_action_fidelity(fid: dict, outpath: Path) -> None:
    if not fid.get("available"):
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    teacher = fid["teacher_arr"]
    student = fid["student_arr"]

    # One color per intervention
    for i, name in enumerate(ACTION_NAMES):
        ax.scatter(
            teacher[:, i], student[:, i],
            s=44, alpha=0.55, edgecolor="white", linewidth=0.4,
            color=COLOR["intervention"][i], label=name,
        )

    # Reference y = x line
    ax.plot([0, 1], [0, 1], color=COLOR["grid"], linestyle="--", linewidth=1.2, zorder=0)

    style_axes(
        ax,
        title="Action vector fidelity — student vs. teacher",
        xlabel="Swarm teacher's recommended intensity",
        ylabel="1B student's predicted intensity",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
        fontsize=8, title="intervention", title_fontsize=9,
    )

    # Annotate with R² and MAE
    txt = (
        f"R² = {fid['pearson']**2:.3f}\n"
        f"MAE = {fid['overall_mae']:.3f}\n"
        f"Top-3 agreement = {fid['top3_agreement']*100:.1f}%\n"
        f"N = {fid['n_examples']}"
    )
    ax.text(
        0.04, 0.96, txt, transform=ax.transAxes, fontsize=10,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor=COLOR["grid"], linewidth=0.6),
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved {outpath}")


# ============================================================================
# Plot 3: Per-intervention comparison
# ============================================================================

def plot_per_intervention(fid: dict, outpath: Path) -> None:
    if not fid.get("available"):
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))

    teacher_means = fid["teacher_arr"].mean(axis=0)
    teacher_stds  = fid["teacher_arr"].std(axis=0)
    student_means = fid["student_arr"].mean(axis=0)
    student_stds  = fid["student_arr"].std(axis=0)

    x = np.arange(len(ACTION_NAMES))
    w = 0.36
    ax.bar(
        x - w/2, teacher_means, w, yerr=teacher_stds,
        label="Swarm teacher", color=COLOR["teacher"], alpha=0.92,
        capsize=3, edgecolor="white", linewidth=0.5,
    )
    ax.bar(
        x + w/2, student_means, w, yerr=student_stds,
        label="1B distilled student", color=COLOR["trained"], alpha=0.92,
        capsize=3, edgecolor="white", linewidth=0.5,
    )

    style_axes(
        ax,
        title="Per-intervention recommended intensity — teacher vs. student",
        xlabel="Intervention",
        ylabel="Recommended intensity (mean ± std)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved {outpath}")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="vishwamitra-1b-lora",
                    help="Path to trained LoRA adapter dir.")
    ap.add_argument("--val", type=Path, default=Path("data/val.jsonl"),
                    help="Path to validation JSONL (for action-vector fidelity).")
    ap.add_argument("--episodes", type=int, default=50,
                    help="Episodes per policy on the env.")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    ap.add_argument("--results", type=Path, default=Path("results.json"))
    ap.add_argument("--skip-env", action="store_true",
                    help="Skip env evaluation (only do action-vector fidelity).")
    ap.add_argument("--skip-fidelity", action="store_true",
                    help="Skip action-vector fidelity (only env eval).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    # --- Env evaluation ---------------------------------------------------
    if not args.skip_env:
        print("[1/2] Evaluating policies on DropoutCommonsEnv...")
        rng = np.random.default_rng(123)
        results = []
        for name, policy_factory in [
            ("random",  lambda: random_policy(rng)),
            ("zero",    lambda: zero_policy()),
            ("trained", lambda: make_trained_policy(args.adapter)),
        ]:
            t0 = time.time()
            r = evaluate_policy(
                name, policy_factory(), args.episodes, args.max_steps,
            )
            r["wall_clock_s"] = time.time() - t0
            results.append(r)
            print(f"    {name:8s}  mean_final={r['mean_final']:+.3f} ± {r['std_final']:.3f}  "
                  f"({r['episodes_solved']}/{args.episodes} solved)")

        plot_reward_curve(results, args.out_dir / "reward_curve.png")

        # Strip array fields before JSON dump
        for r in results:
            r.pop("all_cumulative", None)
            r.pop("mean_cumulative", None)
            r.pop("se_cumulative", None)
            r.pop("final_rewards", None)
        summary["env_eval"] = results

    # --- Action-vector fidelity -------------------------------------------
    if not args.skip_fidelity and args.val.exists():
        print("\n[2/2] Computing action-vector fidelity...")
        fid = evaluate_fidelity(args.adapter, args.val)
        if fid.get("available"):
            plot_action_fidelity(fid, args.out_dir / "action_fidelity.png")
            plot_per_intervention(fid, args.out_dir / "per_intervention.png")
            # Strip arrays before JSON dump
            fid_serialized = {k: v for k, v in fid.items()
                              if k not in ("teacher_arr", "student_arr")}
            summary["fidelity"] = fid_serialized
        else:
            print("  ⚠ Fidelity skipped (model not loadable).")

    # --- Save numeric summary --------------------------------------------
    args.results.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n✓ Saved numbers → {args.results}")
    print(f"✓ Saved plots   → {args.out_dir.resolve()}")
    print()
    print("Now drop into your README:")
    for plot in ("reward_curve.png", "action_fidelity.png", "per_intervention.png"):
        p = args.out_dir / plot
        if p.exists():
            print(f"  ![{plot}]({p})")


if __name__ == "__main__":
    main()
