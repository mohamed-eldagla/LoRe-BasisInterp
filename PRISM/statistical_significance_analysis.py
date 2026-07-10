import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from concept_basis_alignment import analyze_basis_diversity, deduplicate_bases, get_confidence_tier

def compute_global_null_distribution(dim=4096, num_samples=1000, percentile=95, save_path=None):
    """
    Generate global null distribution by computing pairwise cosine similarities
    between random unit vectors.

    This captures: "what cosine similarity would we observe between two arbitrary
    random directions in this high-dimensional space?"

    Args:
        dim: Dimensionality of the space
        num_samples: Number of random vectors to generate
        percentile: Percentile to use as significance threshold
        save_path: Optional path to save the raw null distribution

    Returns:
        threshold: The percentile value (e.g., 95th percentile)
        pairwise_sims: All pairwise similarities (for visualization)
        stats: Dictionary of distribution statistics
    """
    print(f"\n=== Computing Global Null Distribution ===")
    print(f"Generating {num_samples} random unit vectors in {dim}-dimensional space...")

    # Generate random unit vectors
    random_vectors = torch.randn(num_samples, dim)
    random_vectors = F.normalize(random_vectors, p=2, dim=1)

    # Compute all pairwise cosine similarities
    sim_matrix = torch.mm(random_vectors, random_vectors.t())

    # Extract upper triangle (excluding diagonal) to get unique pairs
    upper_tri_indices = torch.triu_indices(num_samples, num_samples, offset=1)
    pairwise_sims = sim_matrix[upper_tri_indices[0], upper_tri_indices[1]]

    # Compute statistics
    mean_sim = pairwise_sims.mean().item()
    std_sim = pairwise_sims.std().item()
    sorted_sims, _ = torch.sort(pairwise_sims)

    percentiles = [90, 95, 99]
    percentile_values = {}
    for p in percentiles:
        idx = int((p / 100.0) * len(sorted_sims))
        percentile_values[p] = sorted_sims[idx].item()

    threshold = percentile_values[percentile]

    stats = {
        "mean": mean_sim,
        "std": std_sim,
        "min": sorted_sims[0].item(),
        "max": sorted_sims[-1].item(),
        **{f"p{p}": v for p, v in percentile_values.items()}
    }

    print(f"\nNull Distribution Statistics:")
    print(f"  Mean: {mean_sim:.4f}")
    print(f"  Std:  {std_sim:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")
    for p in percentiles:
        print(f"  {p}th percentile: {percentile_values[p]:.4f}")

    print(f"\nUsing {percentile}th percentile as significance threshold: τ = {threshold:.4f}")

    # Save if requested
    if save_path:
        torch.save({
            "pairwise_sims": pairwise_sims,
            "stats": stats,
            "threshold": threshold,
            "percentile": percentile,
            "num_samples": num_samples,
            "dim": dim
        }, save_path)
        print(f"Saved null distribution to {save_path}")

    return threshold, pairwise_sims, stats

def plot_null_vs_observed(null_sims, observed_sims, threshold, output_path,
                         title="Null Distribution vs. Observed Similarities"):
    """
    Plot histogram of null distribution with observed similarities overlaid.

    Args:
        null_sims: 1D tensor of null distribution values
        observed_sims: 1D tensor of all observed basis-concept similarities
        threshold: Significance threshold (e.g., 95th percentile)
        output_path: Where to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Convert to numpy
    null_np = null_sims.cpu().numpy()
    obs_np = observed_sims.cpu().numpy()

    # Left plot: Overlaid histograms
    ax1.hist(null_np, bins=50, alpha=0.6, label='Null (random vectors)',
             color='gray', density=True)
    ax1.hist(obs_np, bins=50, alpha=0.6, label='Observed (basis-concept)',
             color='blue', density=True)
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold (95th %ile) = {threshold:.3f}')
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Right plot: Cumulative distributions
    sorted_null = np.sort(null_np)
    sorted_obs = np.sort(obs_np)
    cumulative_null = np.arange(1, len(sorted_null) + 1) / len(sorted_null)
    cumulative_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)

    ax2.plot(sorted_null, cumulative_null, label='Null CDF', color='gray', linewidth=2)
    ax2.plot(sorted_obs, cumulative_obs, label='Observed CDF', color='blue', linewidth=2)
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.3f}')
    ax2.axhline(0.95, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Cosine Similarity', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Distribution Functions', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution comparison plot to {output_path}")
    plt.close()

def analyze_significance(concept_vectors_path, basis_matrix_path,
                        null_samples=1000, percentile=95,
                        deduplicate=True, dedup_threshold=0.95,
                        output_dir="results/significance"):
    """
    Main analysis function with statistical significance testing.

    Performs:
    1. Basis diversity analysis
    2. Optional deduplication
    3. Global null distribution computation
    4. Concept-basis alignment with significance testing
    5. Visualization and reporting
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading concept vectors from {concept_vectors_path}...")
    concept_vectors = torch.load(concept_vectors_path, map_location='cpu', weights_only=True)

    print(f"Loading basis matrix V from {basis_matrix_path}...")
    V_original = torch.load(basis_matrix_path, map_location='cpu', weights_only=True)

    dim = V_original.shape[0]
    concepts = list(concept_vectors.keys())
    C = len(concepts)

    # Step 1: Analyze basis diversity
    diversity_metrics = analyze_basis_diversity(V_original)
    diversity_df = pd.DataFrame([diversity_metrics])
    diversity_df.to_csv(os.path.join(output_dir, "basis_diversity.csv"), index=False)

    # Step 2: Optionally deduplicate
    basis_mapping = None
    if deduplicate and diversity_metrics["mean_similarity"] > 0.80:
        print(f"\nApplying basis deduplication...")
        V, clusters, kept_indices = deduplicate_bases(V_original, similarity_threshold=dedup_threshold)
        basis_mapping = {"clusters": clusters, "kept_indices": kept_indices}

        dedup_info = []
        for i, cluster in enumerate(clusters):
            dedup_info.append({
                "cluster_id": i,
                "representative": f"Basis_{cluster[0]}",
                "cluster_size": len(cluster),
                "members": ", ".join([f"Basis_{b}" for b in cluster])
            })
        dedup_df = pd.DataFrame(dedup_info)
        dedup_df.to_csv(os.path.join(output_dir, "basis_deduplication.csv"), index=False)
    else:
        V = V_original
        print("\nSkipping deduplication (diversity metrics look good or deduplicate=False)")

    B = V.shape[1]

    # Step 3: Compute global null distribution
    threshold, null_sims, null_stats = compute_global_null_distribution(
        dim=dim,
        num_samples=null_samples,
        percentile=percentile,
        save_path=os.path.join(output_dir, "null_distribution.pt")
    )

    # Save null stats to CSV
    null_stats_df = pd.DataFrame([null_stats])
    null_stats_df.to_csv(os.path.join(output_dir, "null_statistics.csv"), index=False)

    # Step 4: Compute basis-concept similarities
    print(f"\n=== Computing Concept-Basis Alignments ===")
    V_norm = F.normalize(V, p=2, dim=0)
    C_mat = torch.stack([concept_vectors[c] for c in concepts], dim=0)
    C_norm = F.normalize(C_mat, p=2, dim=1)
    S = torch.mm(V_norm.t(), C_norm.t())  # [B, C]

    # Step 5: Apply significance test and create report
    report_rows = []
    for b in range(B):
        if basis_mapping:
            original_basis_idx = basis_mapping["kept_indices"][b]
            basis_label = f"Basis_{original_basis_idx}"
        else:
            basis_label = f"Basis_{b}"

        for c_idx, c in enumerate(concepts):
            sim = S[b, c_idx].item()
            tier = get_confidence_tier(sim, threshold)

            report_rows.append({
                "Basis": basis_label,
                "Concept": c,
                "Cosine_Sim": sim,
                "Threshold": threshold,
                "Tier": tier,
                "Significant": tier != "Reject"
            })

    df = pd.DataFrame(report_rows)
    df.to_csv(os.path.join(output_dir, "alignment_report.csv"), index=False)

    # Print summary
    print("\n=== Alignment Summary ===")
    sig_df = df[df["Significant"]].sort_values(by=["Basis", "Cosine_Sim"], ascending=[True, False])

    print(f"\nTotal basis-concept pairs: {len(df)}")
    print(f"Significant alignments: {len(sig_df)} ({100*len(sig_df)/len(df):.1f}%)")
    print(f"\nBreakdown by tier:")
    for tier in ["Excellent", "Good", "Tentative", "Reject"]:
        count = len(df[df["Tier"] == tier])
        print(f"  {tier}: {count} ({100*count/len(df):.1f}%)")

    print(f"\n=== Significant Alignments by Basis ===")
    unique_bases = sorted(df["Basis"].unique())
    for basis_label in unique_bases:
        b_df = sig_df[sig_df["Basis"] == basis_label]
        if len(b_df) > 0:
            print(f"\n{basis_label}:")
            for _, row in b_df.iterrows():
                print(f"  - {row['Concept']}: {row['Cosine_Sim']:>6.3f} ({row['Tier']})")
        else:
            print(f"\n{basis_label}: No significant alignments (all below threshold)")

    # Step 6: Visualizations
    print(f"\n=== Generating Visualizations ===")

    # Plot null vs observed distributions
    observed_sims = S.flatten()
    plot_null_vs_observed(
        null_sims, observed_sims, threshold,
        os.path.join(output_dir, "null_vs_observed.png")
    )

    # Summary statistics comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_data = {
        "Distribution": ["Null (random)", "Observed (basis-concept)"],
        "Mean": [null_stats["mean"], observed_sims.mean().item()],
        "Std": [null_stats["std"], observed_sims.std().item()],
        "Max": [null_stats["max"], observed_sims.max().item()],
        "95th %ile": [null_stats["p95"], torch.quantile(observed_sims, 0.95).item()]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, "distribution_comparison.csv"), index=False)

    # Bar plot comparison
    x = np.arange(len(comparison_data["Distribution"]))
    width = 0.15
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    for attribute in ["Mean", "Std", "Max", "95th %ile"]:
        offset = width * multiplier
        ax.bar(x + offset, comparison_data[attribute], width, label=attribute)
        multiplier += 1

    ax.set_xlabel('Distribution Type', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Statistical Comparison: Null vs. Observed', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_data["Distribution"])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistics_comparison.png"), dpi=300)
    plt.close()

    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to {output_dir}/")
    print(f"  - alignment_report.csv: Detailed alignments with significance")
    print(f"  - basis_diversity.csv: Basis diversity metrics")
    print(f"  - null_statistics.csv: Null distribution statistics")
    print(f"  - distribution_comparison.csv: Statistical comparison")
    print(f"  - null_vs_observed.png: Distribution comparison plot")
    print(f"  - statistics_comparison.png: Bar chart comparison")
    if basis_mapping:
        print(f"  - basis_deduplication.csv: Cluster information")

if __name__ == "__main__":
    analyze_significance(
        concept_vectors_path="data/prism/concept_vectors.pt",
        basis_matrix_path="checkpoints/checkpoints/PRISM_V_lore_K_10_alpha_10000.0.pt",
        null_samples=1000,
        percentile=95,
        deduplicate=True,
        dedup_threshold=0.95,
        output_dir="results/K_10_significance"
    )
