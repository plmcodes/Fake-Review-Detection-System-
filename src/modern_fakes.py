"""Phase 8 - do the detectors still catch fakes written by a MODERN language model?

Salminen's fakes come from GPT-2 (2019). This script builds a small set of fake reviews written by a
recent, free, open-weights instruction model and measures each trained model's DETECTION RATE on it
(share flagged as fake with the frozen threshold = recall, since every review here is fake).

    python -m src.modern_fakes generate --n 200          # writes data/raw/modern_fakes.csv
    python -m src.modern_fakes evaluate                  # scores it with every seed-42 checkpoint

No paid API is used. Options:
  (a) write data/raw/modern_fakes.csv yourself (one column `text`) and only run `evaluate`, or
  (b) `generate` with a small Hugging Face model (default Qwen/Qwen2.5-1.5B-Instruct, Apache-2.0; fits a
      T4, and also runs - slowly - on CPU). Use --model to pick another open model.

Results: rows with eval_dataset=modern_fakes in results/runs.csv and results/tables/modern_fakes.md.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd

from .config import load_config, resolve_path, set_seed
from .utils import get_logger, log_run

log = get_logger("modern_fakes")

# Same 10 Amazon categories as the Salminen data, so the domain matches.
CATEGORIES = {
    "Home_and_Kitchen": ["blender", "non-stick frying pan", "memory foam pillow", "coffee grinder", "knife set"],
    "Sports_and_Outdoors": ["yoga mat", "hiking backpack", "resistance bands", "water bottle", "camping lantern"],
    "Electronics": ["bluetooth earbuds", "USB-C charger", "portable SSD", "HDMI cable", "wireless mouse"],
    "Movies_and_TV": ["a crime drama box set", "an animated family movie", "a nature documentary"],
    "Tools_and_Home_Improvement": ["cordless drill", "stud finder", "LED shop light", "tape measure"],
    "Pet_Supplies": ["cat scratching post", "dog chew toy", "automatic pet feeder", "dog harness"],
    "Kindle_Store": ["a romance novel", "a thriller e-book", "a self-help e-book"],
    "Books": ["a cookbook", "a historical fiction novel", "a children's picture book"],
    "Toys_and_Games": ["a 1000-piece jigsaw puzzle", "a board game for families", "a remote control car"],
    "Clothing_Shoes_and_Jewelry": ["running shoes", "a winter jacket", "a silver necklace", "cotton t-shirts"],
}
STYLES = ["casual and short", "enthusiastic", "detailed and practical", "slightly critical but positive overall",
          "written like a busy parent", "matter-of-fact"]


def build_prompts(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        cat = rng.choice(list(CATEGORIES))
        rating = rng.choices([5, 4, 3, 2, 1], weights=[45, 25, 12, 8, 10])[0]
        product, style = rng.choice(CATEGORIES[cat]), rng.choice(STYLES)
        words = rng.choice([25, 40, 60, 90])
        prompt = (f"Write a realistic {rating}-star Amazon customer review of {product}. "
                  f"Tone: {style}. About {words} words. Write only the review text, as a real customer would, "
                  "with no title and no star rating.")
        items.append({"review_id": f"modern_{i}", "category": cat, "rating": float(rating), "prompt": prompt})
    return items


def generate(n: int, model_name: str, seed: int, batch_size: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    items = build_prompts(n, seed)
    out_path = resolve_path("data/raw/modern_fakes.csv")
    for start in range(0, n, batch_size):
        batch = items[start : start + batch_size]
        chats = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                         add_generation_prompt=True) for it in batch]
        enc = tok(chats, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.9, top_p=0.95,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for it, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
            it["text"] = tok.decode(g, skip_special_tokens=True).strip().strip('"')
            it["generator"] = model_name
        log.info("generated %d/%d", min(start + batch_size, n), n)
        # save as we go, so an interrupted run keeps its progress
        pd.DataFrame([it for it in items if "text" in it]).drop(columns="prompt").to_csv(out_path, index=False)
    log.info("wrote %s", out_path)


def evaluate(seed: int) -> pd.DataFrame:
    """Detection rate of every seed-`seed` full-data Salminen model on data/raw/modern_fakes.csv."""
    from .evaluate import PRETTY, _order_key, compute_metrics, load_runs, write_table
    from .features import load_bundle
    from .models import load_runner
    from .utils import load_json

    cfg = load_config()
    rows = []
    for meta_path in sorted(resolve_path(cfg["paths"]["checkpoints"]).glob(f"*__salminen__seed{seed}/meta.json")):
        meta = load_json(meta_path)
        runner, meta = load_runner(meta_path.parent)
        data = load_bundle("modern_fakes", meta["config"], as_eval_set=True)
        idx = data.idx("test")
        scores = runner.score(data, idx)
        m = compute_metrics(data.y[idx], scores, meta["threshold"])
        log_run({"run_id": meta["run_id"], "model": meta["model"], "family": meta["family"],
                 "needs_labels": meta["needs_labels"], "train_dataset": "salminen", "eval_dataset": "modern_fakes",
                 "split": "test", "tag": meta.get("tag", ""), "config_hash": meta["config_hash"], "seed": meta["seed"],
                 "sample": "", "recall": m["recall"], "threshold": m["threshold"], "tp": m["tp"], "fn": m["fn"],
                 "n_eval": len(idx), "checkpoint": str(meta_path.parent.relative_to(resolve_path("."))),
                 "notes": "all reviews are fake: only recall (= detection rate) is meaningful"}, cfg["paths"]["runs_csv"])
        rows.append({"model": meta["model"], "run_id": meta["run_id"], "detection_rate": m["recall"],
                     "median_score": float(np.median(scores))})
        log.info("%-14s detection rate %.3f", meta["model"], m["recall"])

    runs = load_runs(cfg)
    gpt2 = runs[(runs["eval_dataset"] == "salminen") & (runs["split"] == "test") & (runs["seed"] == seed)
                & (runs["sample"] == "") & (runs["tag"] == "")].set_index("run_id")["recall"]
    table = pd.DataFrame(rows)
    if table.empty:
        log.warning("no seed-%d checkpoints found", seed)
        return table
    table["GPT-2 fakes (Salminen test) recall"] = table["run_id"].map(gpt2).map(lambda v: f"{v:.3f}" if pd.notna(v) else "TBD")
    table["Modern-LLM fakes detection rate"] = table["detection_rate"].map(lambda v: f"{v:.3f}")
    table["Model"] = table["model"].map(lambda m: PRETTY.get(m, m))
    table = table.iloc[sorted(range(len(table)), key=lambda i: _order_key(table["model"].iloc[i]))]
    gen = pd.read_csv(resolve_path("data/raw/modern_fakes.csv")).get("generator", pd.Series(["hand-written"]))
    write_table(table[["Model", "GPT-2 fakes (Salminen test) recall", "Modern-LLM fakes detection rate"]],
                "modern_fakes", cfg, "Phase 8: detection of modern-LLM fakes",
                f"{len(data.df)} fake reviews generated by `{gen.iloc[0]}`; seed-{seed} models, frozen thresholds. "
                "Detection rate = share flagged as fake.")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("evaluate")
    e.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        generate(args.n, args.model, args.seed, args.batch_size)
    else:
        evaluate(args.seed)


if __name__ == "__main__":
    main()
