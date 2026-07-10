"""
Test script for basis deduplication and diversity analysis.
Creates synthetic data to verify the implementation works correctly.
"""

import torch
import torch.nn.functional as F
from concept_basis_alignment import analyze_basis_diversity, deduplicate_bases

def create_test_bases(scenario="collapsed"):
    """
    Create synthetic basis matrices for testing.

    Args:
        scenario: "collapsed", "moderate", or "diverse"

    Returns:
        V: Basis matrix [4096, K]
        ground_truth_clusters: Expected cluster assignments
    """
    dim = 4096

    if scenario == "collapsed":
        # Simulate severe collapse: 7 bases, all nearly identical
        print("\n=== Test Case: Severe Collapse (like your colleague's model) ===")
        base_vector = torch.randn(dim)
        base_vector = F.normalize(base_vector, p=2, dim=0)

        # Create 7 bases with tiny random perturbations
        bases = []
        for i in range(7):
            perturbed = base_vector + 0.01 * torch.randn(dim)
            perturbed = F.normalize(perturbed, p=2, dim=0)
            bases.append(perturbed)

        V = torch.stack(bases, dim=1)
        ground_truth_clusters = [[0, 1, 2, 3, 4, 5, 6]]  # All in one cluster

    elif scenario == "moderate":
        # Simulate moderate collapse: 10 bases, 3 groups
        print("\n=== Test Case: Moderate Collapse (3 groups) ===")
        dim = 4096

        # Group 1: bases 0-3 (similar)
        base1 = F.normalize(torch.randn(dim), p=2, dim=0)
        group1 = [base1 + 0.05 * torch.randn(dim) for _ in range(4)]
        group1 = [F.normalize(b, p=2, dim=0) for b in group1]

        # Group 2: bases 4-6 (similar)
        base2 = F.normalize(torch.randn(dim), p=2, dim=0)
        group2 = [base2 + 0.05 * torch.randn(dim) for _ in range(3)]
        group2 = [F.normalize(b, p=2, dim=0) for b in group2]

        # Group 3: bases 7-9 (similar)
        base3 = F.normalize(torch.randn(dim), p=2, dim=0)
        group3 = [base3 + 0.05 * torch.randn(dim) for _ in range(3)]
        group3 = [F.normalize(b, p=2, dim=0) for b in group3]

        V = torch.stack(group1 + group2 + group3, dim=1)
        ground_truth_clusters = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

    else:  # diverse
        # Simulate good diversity: 10 random bases
        print("\n=== Test Case: Good Diversity (10 independent bases) ===")
        bases = [F.normalize(torch.randn(dim), p=2, dim=0) for _ in range(10)]
        V = torch.stack(bases, dim=1)
        ground_truth_clusters = [[i] for i in range(10)]  # Each basis is its own cluster

    return V, ground_truth_clusters

def test_diversity_analysis():
    """Test diversity analysis on different scenarios."""

    for scenario in ["collapsed", "moderate", "diverse"]:
        V, expected_clusters = create_test_bases(scenario)
        metrics = analyze_basis_diversity(V)

        print(f"\nExpected behavior:")
        if scenario == "collapsed":
            print("  - Mean similarity should be > 0.95")
            print("  - Should detect severe collapse")
            assert metrics["mean_similarity"] > 0.95, "Failed to detect collapsed bases"
            print("✓ Test passed!")

        elif scenario == "moderate":
            print("  - Mean similarity should be > 0.80")
            print("  - Should detect moderate collapse")
            assert 0.80 < metrics["mean_similarity"] < 0.95, "Failed to detect moderate collapse"
            print("✓ Test passed!")

        else:  # diverse
            print("  - Mean similarity should be < 0.50")
            print("  - Should show good diversity")
            assert metrics["mean_similarity"] < 0.50, "False positive: flagged diverse bases as collapsed"
            print("✓ Test passed!")

        print("\n" + "="*70)

def test_deduplication():
    """Test deduplication on collapsed bases."""

    print("\n" + "="*70)
    print("Testing Deduplication")
    print("="*70)

    # Test on collapsed case
    V, expected_clusters = create_test_bases("collapsed")
    V_unique, clusters, kept_indices = deduplicate_bases(V, similarity_threshold=0.95)

    print(f"\nValidation:")
    print(f"  - Expected 1 unique basis, got {V_unique.shape[1]}")
    assert V_unique.shape[1] == 1, "Deduplication failed: should have 1 unique basis"
    print("  ✓ Correct number of unique bases")

    print(f"  - Expected 1 cluster with 7 members, got {len(clusters)} cluster(s)")
    assert len(clusters) == 1, "Deduplication failed: should have 1 cluster"
    assert len(clusters[0]) == 7, "Deduplication failed: cluster should have 7 members"
    print("  ✓ Correct clustering")

    # Test on moderate case
    V, expected_clusters = create_test_bases("moderate")
    V_unique, clusters, kept_indices = deduplicate_bases(V, similarity_threshold=0.95)

    print(f"\nModerate collapse test:")
    print(f"  - Expected 3 unique bases, got {V_unique.shape[1]}")
    assert V_unique.shape[1] == 3, "Deduplication failed: should have 3 unique bases"
    print("  ✓ Correct number of unique bases")

    print(f"  - Expected 3 clusters, got {len(clusters)}")
    assert len(clusters) == 3, "Deduplication failed: should have 3 clusters"
    print("  ✓ Correct number of clusters")

    # Test on diverse case (should keep all)
    V, expected_clusters = create_test_bases("diverse")
    V_unique, clusters, kept_indices = deduplicate_bases(V, similarity_threshold=0.95)

    print(f"\nDiverse case test:")
    print(f"  - Expected 10 unique bases, got {V_unique.shape[1]}")
    assert V_unique.shape[1] == 10, "Deduplication failed: should keep all 10 bases"
    print("  ✓ No false positives (all bases kept)")

    print("\n" + "="*70)
    print("All deduplication tests passed!")
    print("="*70)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING BASIS DEDUPLICATION AND DIVERSITY ANALYSIS")
    print("="*70)

    test_diversity_analysis()
    test_deduplication()

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print("\nThe implementation is ready to use on real data.")
    print("Run: python PRISM/statistical_significance_analysis.py")
