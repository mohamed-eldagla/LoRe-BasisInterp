# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Alpha sweep for basis collapse.

Hypothesis: the PRISM bases collapse (pairwise cosine similarity > 0.99) because
alpha is set unusually high (1e4). The regularization term in LoRe_regularized
(utils.py) pushes EVERY basis column toward the single reference direction V_sft
via cosine similarity. When alpha dominates the preference-fitting NLL, all columns
converge onto that one direction. Other datasets (PersonalLLM, RedditTLDR) use alpha=0.

This script trains the LoRe bases for PRISM across a range of alpha values at a
fixed rank K, then measures basis collapse for each. It reuses:
  - solve_regularized_simplex (utils.py) -- the exact training routine PRISM uses
    via run_regularized, minus the hardcoded checkpoint save path and eval.
  - analyze_basis_diversity (concept_basis_alignment.py) -- collapse diagnostics.

Outputs:
  - Trained basis matrix per alpha: {output_dir}/PRISM_V_lore_K_{K}_alpha_{alpha}.pt
  - Collapse summary CSV: {output_dir}/alpha_collapse_summary.csv
  - Collapse-vs-alpha plot: {output_dir}/alpha_collapse.png
"""

import os
import sys
import argparse
from collections import defaultdict

import torch
import pandas as pd
import matplotlib.pyplot as plt

# Make sibling modules importable (utils.py lives one level up; concept_basis_alignment
# lives in this directory).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)

from utils import solve_regularized_simplex
from concept_basis_alignment import analyze_basis_diversity


def group_embeddings_by_user(train_embeddings, device):
    """
    Group the chosen-minus-rejected embedding differences by seen training user.

    Mirrors the grouping in train_basis.py, but only builds the train_seen split,
    which is all that is needed to fit the reward bases.
    """
    grouped = defaultdict(lambda: {"embeddings": []})
    for example in train_embeddings:
        extra_info = example.get("extra_info", {})
        if extra_info.get("seen") is True and extra_info.get("split") == "train":
            user_id = extra_info.get("user_id")
            if user_id:
                chosen = torch.tensor(extra_info["chosen_conv_embedding"], dtype=torch.float32, device=device)
                rejected = torch.tensor(extra_info["rejected_conv_embedding"], dtype=torch.float32, device=device)
                grouped[user_id]["embeddings"].append(chosen - rejected)

    train_seen = []
    count = 0
    for user_id in sorted(grouped.keys()):
        count += len(grouped[user_id]["embeddings"])
        train_seen.append(torch.stack(grouped[user_id]["embeddings"]))
    print(f"Grouped {count} preference pairs across {len(train_seen)} seen users.")
    return train_seen


def load_reference_direction(model_name, device):
    """
    Extract V_sft: the backbone reward model's single reward direction, taken from
    the first column of the last linear layer. This is the regularization target.
    Identical to the extraction in train_basis.py.
    """
    from transformers import AutoModel

    print(f"Loading reference direction from {model_name}...")
    rm = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        num_labels=1,
    )
    last_linear_layer = None
    for _, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module
    V_ref = last_linear_layer.weight[:, 0].to(device).to(torch.float32).reshape(-1, 1)
    # Free the backbone; only V_ref is needed for training.
    del rm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return V_ref


def scan_alpha(
    embeddings_path="data/prism/train_embeddings.pkl",
    model_name="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    alpha_list=(0, 0.01, 0.1, 1, 10000),
    K=10,
    num_iterations=20000,
    learning_rate=0.5,
    collapse_threshold=0.95,
    output_dir="results/alpha_scan",
    device=None,
):
    """
    Train PRISM bases across alpha_list at fixed rank K and measure collapse.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and group training embeddings.
    print(f"Loading training embeddings from {embeddings_path}...")
    train_embeddings = torch.load(embeddings_path)
    train_seen = group_embeddings_by_user(train_embeddings, device)

    # 2. Load the reference direction used by the regularizer.
    V_ref = load_reference_direction(model_name, device)

    # 3. Sweep alpha.
    summary_rows = []
    for alpha in alpha_list:
        print("\n" + "=" * 70)
        print(f"Training bases: alpha={alpha}, K={K}")
        print("=" * 70)

        W, V = solve_regularized_simplex(
            V_ref, alpha, train_seen, K,
            num_iterations=num_iterations, learning_rate=learning_rate,
        )
        V = V.detach().cpu()

        # Save the trained basis so it can be reused by the alignment pipeline.
        basis_path = os.path.join(output_dir, f"PRISM_V_lore_K_{K}_alpha_{alpha}.pt")
        torch.save(V, basis_path)
        print(f"Saved basis matrix to {basis_path}")

        # Measure collapse. Note: the training routine drops unused directions, so
        # the returned V may have fewer than K columns.
        num_effective = V.shape[1]
        if num_effective < 2:
            print(f"WARNING: only {num_effective} basis column(s) survived pruning; "
                  f"cannot compute pairwise similarity.")
            summary_rows.append({
                "alpha": alpha,
                "K_requested": K,
                "num_effective_bases": num_effective,
                "mean_similarity": float("nan"),
                "max_similarity": float("nan"),
                "min_similarity": float("nan"),
                "collapsed": num_effective <= 1,
            })
            continue

        metrics = analyze_basis_diversity(V)
        summary_rows.append({
            "alpha": alpha,
            "K_requested": K,
            "num_effective_bases": num_effective,
            "mean_similarity": metrics["mean_similarity"],
            "max_similarity": metrics["max_similarity"],
            "min_similarity": metrics["min_similarity"],
            "collapsed": metrics["mean_similarity"] > collapse_threshold,
        })

    # 4. Write summary CSV.
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "alpha_collapse_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 70)
    print("ALPHA SWEEP SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # 5. Plot mean pairwise similarity vs alpha (log-scale x, with alpha=0 handled).
    plot_alphas = summary_df["alpha"].tolist()
    mean_sims = summary_df["mean_similarity"].tolist()
    labels = ["0" if a == 0 else str(a) for a in plot_alphas]
    x = list(range(len(plot_alphas)))

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_sims, marker="o", linestyle="-")
    plt.axhline(collapse_threshold, color="red", linestyle="--",
                label=f"Collapse threshold ({collapse_threshold})")
    plt.xticks(x, labels)
    plt.xlabel("alpha")
    plt.ylabel("Mean pairwise cosine similarity")
    plt.title(f"Basis Collapse vs. Alpha (K={K})")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "alpha_collapse.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nSaved summary to {summary_csv}")
    print(f"Saved plot to {plot_path}")
    print(f"Trained bases saved in {output_dir}/ (one .pt per alpha)")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep alpha and test for basis collapse on PRISM.")
    parser.add_argument("--embeddings_path", default="data/prism/train_embeddings.pkl")
    parser.add_argument("--model_name", default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0, 0.01, 0.1, 1, 10000])
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=20000)
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--collapse_threshold", type=float, default=0.95)
    parser.add_argument("--output_dir", default="results/alpha_scan")
    args = parser.parse_args()

    scan_alpha(
        embeddings_path=args.embeddings_path,
        model_name=args.model_name,
        alpha_list=args.alphas,
        K=args.K,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        collapse_threshold=args.collapse_threshold,
        output_dir=args.output_dir,
    )
