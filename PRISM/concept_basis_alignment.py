import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_null_threshold(concept_vector, num_samples=1000, percentile=95):
    """
    Computes the null distribution of cosine similarities by generating random vectors in the same space.
    Returns the given percentile as the significance threshold (tau_c).
    """
    dim = concept_vector.shape[0]
    # Generate random unit vectors
    random_vectors = torch.randn(num_samples, dim, device=concept_vector.device)
    random_vectors = F.normalize(random_vectors, p=2, dim=1)
    
    # Normalize concept vector
    concept_normalized = F.normalize(concept_vector.unsqueeze(0), p=2, dim=1)
    
    # Compute cosine similarities
    similarities = torch.mm(random_vectors, concept_normalized.t()).squeeze()
    
    # Sort and find percentile
    sorted_sims, _ = torch.sort(similarities)
    idx = int((percentile / 100.0) * num_samples)
    
    return sorted_sims[idx].item()

def analyze_basis_diversity(V):
    """
    Compute and report basis diversity metrics.

    Returns diagnostics about whether bases are distinct or collapsed.
    """
    V_norm = F.normalize(V, p=2, dim=0)
    B = V.shape[1]

    # Compute all pairwise similarities
    basis_sim_matrix = torch.mm(V_norm.t(), V_norm)

    # Get upper triangle (excluding diagonal)
    upper_tri_indices = torch.triu_indices(B, B, offset=1)
    pairwise_sims = basis_sim_matrix[upper_tri_indices[0], upper_tri_indices[1]]

    mean_sim = pairwise_sims.mean().item()
    max_sim = pairwise_sims.max().item()
    min_sim = pairwise_sims.min().item()

    print("\n=== Basis Diversity Analysis ===")
    print(f"Number of bases: {B}")
    print(f"Mean pairwise similarity: {mean_sim:.4f}")
    print(f"Max pairwise similarity: {max_sim:.4f}")
    print(f"Min pairwise similarity: {min_sim:.4f}")

    # Collapse detection
    if mean_sim > 0.95:
        print("WARNING: SEVERE basis collapse detected (mean sim > 0.95)")
        print("   All bases are nearly identical. Interpretability is compromised.")
        print("   Recommendation: Use basis deduplication before alignment analysis.")
    elif mean_sim > 0.80:
        print("WARNING: Moderate basis collapse detected (mean sim > 0.80)")
        print("   Many bases are highly similar. Consider deduplication.")
    else:
        print("Bases show good diversity")

    return {
        "mean_similarity": mean_sim,
        "max_similarity": max_sim,
        "min_similarity": min_sim,
        "num_bases": B,
        "collapse_detected": mean_sim > 0.95,
        "basis_sim_matrix": basis_sim_matrix
    }

def deduplicate_bases(V, similarity_threshold=0.95):
    """
    Cluster highly similar bases and keep only one representative per cluster.

    Args:
        V: Basis matrix [dim, B]
        similarity_threshold: Cosine similarity above which bases are considered duplicates

    Returns:
        V_unique: Deduplicated basis matrix [dim, B_unique]
        clusters: List of lists, where each sublist contains indices of bases in that cluster
        kept_indices: Indices of bases that were kept as representatives
    """
    V_norm = F.normalize(V, p=2, dim=0)

    # Compute pairwise basis similarities
    basis_sim_matrix = torch.mm(V_norm.t(), V_norm)

    # Greedy clustering: keep first basis, remove all similar ones, repeat
    clusters = []
    remaining = set(range(V.shape[1]))

    while remaining:
        # Pick next representative (lowest index among remaining)
        rep = min(remaining)
        cluster = [rep]
        remaining.remove(rep)

        # Find all bases similar to this representative
        for b in list(remaining):
            if basis_sim_matrix[rep, b] > similarity_threshold:
                cluster.append(b)
                remaining.remove(b)

        clusters.append(cluster)

    # Keep only representatives
    kept_indices = [cluster[0] for cluster in clusters]
    V_unique = V[:, kept_indices]

    print(f"\n=== Basis Deduplication (threshold={similarity_threshold}) ===")
    print(f"Original bases: {V.shape[1]}")
    print(f"Unique bases: {len(kept_indices)}")
    print(f"Removed: {V.shape[1] - len(kept_indices)}")

    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            print(f"\nCluster {i}: Kept Basis_{cluster[0]}, removed {len(cluster)-1} duplicates:")
            print(f"  Duplicates: {[f'Basis_{b}' for b in cluster[1:]]}")

    return V_unique, clusters, kept_indices

def get_confidence_tier(similarity, tau_c):
    if similarity < tau_c:
        return "Reject"
    elif similarity < 0.15:
        return "Tentative"
    elif similarity < 0.30:
        return "Good"
    else:
        return "Excellent"

def run_alignment_analysis(concept_vectors_path, basis_matrix_path, output_dir="results",
                          deduplicate=True, dedup_threshold=0.95):
    print(f"Loading concept vectors from {concept_vectors_path}...")
    concept_vectors = torch.load(concept_vectors_path, map_location='cpu', weights_only=True)

    print(f"Loading basis matrix V from {basis_matrix_path}...")
    V_original = torch.load(basis_matrix_path, map_location='cpu', weights_only=True)

    os.makedirs(output_dir, exist_ok=True)

    # Analyze basis diversity
    diversity_metrics = analyze_basis_diversity(V_original)

    # Save diversity metrics
    diversity_df = pd.DataFrame([diversity_metrics])
    diversity_df.to_csv(os.path.join(output_dir, "basis_diversity.csv"), index=False)

    # Optionally deduplicate bases
    basis_mapping = None
    if deduplicate and diversity_metrics["mean_similarity"] > 0.80:
        print(f"\nApplying basis deduplication...")
        V, clusters, kept_indices = deduplicate_bases(V_original, similarity_threshold=dedup_threshold)

        # Create mapping from original to deduplicated indices
        basis_mapping = {"clusters": clusters, "kept_indices": kept_indices}

        # Save deduplication info
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
    
    concepts = list(concept_vectors.keys())
    C = len(concepts)
    
    # 1. Compute per-concept thresholds
    print("Computing null distributions...")
    tau_c_dict = {}
    for c in concepts:
        tau_c_dict[c] = compute_null_threshold(concept_vectors[c])
        print(f"  {c}: τ_95 = {tau_c_dict[c]:.4f}")
        
    # 2. Compute similarity matrix [B, C]
    # We normalize both to compute cosine similarity
    V_norm = F.normalize(V, p=2, dim=0) # [4096, B]
    
    # Stack concept vectors
    C_mat = torch.stack([concept_vectors[c] for c in concepts], dim=0) # [C, 4096]
    C_norm = F.normalize(C_mat, p=2, dim=1) # [C, 4096]
    
    # Cosine similarity matrix S: [B, C]
    # V_norm.t() is [B, 4096], C_norm.t() is [4096, C]
    # S = V_norm.t() @ C_norm.t() -> shape [B, C]
    S = torch.mm(V_norm.t(), C_norm.t())
    
    # 3. Apply tiered rubric and create report
    report_rows = []
    
    for b in range(B):
        basis_sims = S[b, :]
        # If we deduplicated, map back to original basis index
        if basis_mapping:
            original_basis_idx = basis_mapping["kept_indices"][b]
            basis_label = f"Basis_{original_basis_idx}"
        else:
            basis_label = f"Basis_{b}"

        for c_idx, c in enumerate(concepts):
            sim = basis_sims[c_idx].item()
            tau_c = tau_c_dict[c]
            tier = get_confidence_tier(sim, tau_c)

            report_rows.append({
                "Basis": basis_label,
                "Concept": c,
                "Cosine_Sim": sim,
                "Threshold": tau_c,
                "Tier": tier
            })
            
    df = pd.DataFrame(report_rows)
    df.to_csv(os.path.join(output_dir, "alignment_report.csv"), index=False)
    
    # Print summary
    print("\n=== Significant Alignments ===")
    sig_df = df[df["Tier"] != "Reject"].sort_values(by=["Basis", "Cosine_Sim"], ascending=[True, False])
    unique_bases = sorted(df["Basis"].unique())
    for basis_label in unique_bases:
        b_df = sig_df[sig_df["Basis"] == basis_label]
        if len(b_df) > 0:
            print(f"\n{basis_label}:")
            for _, row in b_df.iterrows():
                print(f"  - {row['Concept']}: {row['Cosine_Sim']:>6.3f} ({row['Tier']})")
        else:
            print(f"\n{basis_label}: No significant alignments.")
            
    # 4. Heatmap visualization
    plt.figure(figsize=(12, 8))
    S_np = S.cpu().numpy()

    # Use original basis labels in heatmap
    if basis_mapping:
        yticklabels = [f"Basis {basis_mapping['kept_indices'][b]}" for b in range(B)]
    else:
        yticklabels = [f"Basis {b}" for b in range(B)]

    ax = sns.heatmap(S_np, xticklabels=concepts, yticklabels=yticklabels,
                     cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Cosine Similarity: Concept Vectors vs LoRe Bases")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_heatmap.png"), dpi=300)

    print(f"\nAnalysis complete! Check {output_dir}/ for:")
    print(f"  - alignment_report.csv: Per-basis concept alignments")
    print(f"  - basis_diversity.csv: Diversity metrics")
    if basis_mapping:
        print(f"  - basis_deduplication.csv: Cluster information")
    print(f"  - similarity_heatmap.png: Visualization")

if __name__ == "__main__":
    # Point this to a specific checkpoint you want to evaluate
    run_alignment_analysis(
        concept_vectors_path="data/prism/concept_vectors.pt",
        basis_matrix_path="checkpoints/checkpoints/PRISM_V_lore_K_10_alpha_10000.0.pt",
        output_dir="results/K_10",
        deduplicate=True,
        dedup_threshold=0.95
    )
