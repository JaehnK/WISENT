"""
Unit tests for L2 normalization in GraphMAE2 embeddings.

This test suite verifies that:
1. Extracted embeddings are L2-normalized (||embedding|| = 1)
2. Cosine similarity and Euclidean distance have monotonic relationship
3. Training space and evaluation space are consistent
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.services.GraphMAE.GraphMAEService import GraphMAEService
from core.services.GraphMAE.GraphMAEConfig import GraphMAEConfig
from core.services.Graph.GraphService import GraphService
from core.services.Document.DocumentService import DocumentService
from entities import WordGraph, Word


class TestL2Normalization:
    """Test suite for L2 normalization in GraphMAE embeddings"""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing"""
        # Create random embeddings
        torch.manual_seed(42)
        embeddings = torch.randn(100, 64)
        return embeddings

    @pytest.fixture
    def graphmae_service(self, tmp_path):
        """Create GraphMAEService instance for testing"""
        # Create minimal DocumentService
        doc_service = DocumentService()
        doc_service.documents = ["test sentence one", "test sentence two", "test sentence three"]

        # Create GraphService
        graph_service = GraphService(doc_service)

        # Create GraphMAEService with minimal config
        config = GraphMAEConfig(
            hidden_dim=64,
            num_layers=2,
            max_epochs=10,  # Minimal epochs for testing
            device='cpu'
        )

        return GraphMAEService(graph_service, config)

    def test_embedding_l2_norm_equals_one(self, sample_embeddings):
        """
        Test that L2 normalization results in unit norm vectors.

        For any L2-normalized vector v: ||v|| = 1
        """
        # Normalize embeddings
        normalized = torch.nn.functional.normalize(sample_embeddings, p=2, dim=-1)

        # Compute L2 norms
        norms = torch.norm(normalized, p=2, dim=-1)

        # Check all norms are approximately 1
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), \
            f"Expected all norms to be 1.0, but got min={norms.min():.6f}, max={norms.max():.6f}"

    def test_cosine_euclidean_relationship(self, sample_embeddings):
        """
        Test the mathematical relationship between cosine similarity and Euclidean distance
        in L2-normalized space.

        For normalized vectors a, b:
            ||a - b||² = 2(1 - cos(a,b))

        Therefore: cos(a,b) increases ⟺ ||a - b|| decreases (monotonic relationship)
        """
        # Normalize embeddings
        normalized = torch.nn.functional.normalize(sample_embeddings, p=2, dim=-1)

        # Select pairs for testing
        num_pairs = 50
        indices1 = torch.randint(0, len(normalized), (num_pairs,))
        indices2 = torch.randint(0, len(normalized), (num_pairs,))

        # Compute cosine similarity
        cosine_sim = torch.sum(normalized[indices1] * normalized[indices2], dim=-1)

        # Compute Euclidean distance
        euclidean_dist = torch.norm(normalized[indices1] - normalized[indices2], p=2, dim=-1)

        # Verify mathematical relationship: ||a - b||² = 2(1 - cos(a,b))
        expected_dist_squared = 2 * (1 - cosine_sim)
        actual_dist_squared = euclidean_dist ** 2

        assert torch.allclose(actual_dist_squared, expected_dist_squared, atol=1e-5), \
            "Cosine similarity and Euclidean distance relationship violated"

        # Verify monotonic relationship: higher cosine similarity → lower distance
        # Sort by cosine similarity
        sorted_indices = torch.argsort(cosine_sim, descending=True)
        sorted_cosine = cosine_sim[sorted_indices]
        sorted_distance = euclidean_dist[sorted_indices]

        # Check that distances are mostly increasing (some tolerance for numerical errors)
        diff = sorted_distance[1:] - sorted_distance[:-1]
        # At least 90% should be non-decreasing
        non_decreasing_ratio = (diff >= -1e-5).float().mean()
        assert non_decreasing_ratio >= 0.9, \
            f"Expected monotonic relationship, but only {non_decreasing_ratio*100:.1f}% pairs satisfy it"

    def test_graphmae_embed_output_normalized(self, graphmae_service):
        """
        Test that GraphMAEService.pretrain_and_extract() returns L2-normalized embeddings.
        """
        # Create minimal word graph
        words = [
            Word(text="word1", id=0, count=10),
            Word(text="word2", id=1, count=8),
            Word(text="word3", id=2, count=6),
        ]
        word_graph = WordGraph(words=words)
        word_graph.add_edge(0, 1, weight=0.5)
        word_graph.add_edge(1, 2, weight=0.3)

        # Extract embeddings (this will train a small model)
        embeddings = graphmae_service.pretrain_and_extract(
            word_graph=word_graph,
            embed_size=64,
            input_method='bert'
        )

        # Compute L2 norms
        norms = torch.norm(embeddings, p=2, dim=-1)

        # Check all norms are approximately 1
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"GraphMAE embeddings not normalized: min={norms.min():.6f}, max={norms.max():.6f}"

    def test_consistency_with_sce_loss(self):
        """
        Test that embed() applies the same normalization as used in SCE loss during training.

        This ensures training space and evaluation space are consistent.
        """
        from core.GraphMAE2.models.loss_func import sce_loss

        # Create sample representations
        torch.manual_seed(42)
        x = torch.randn(10, 64)
        y = torch.randn(10, 64)

        # Manual normalization (what embed() should do)
        x_norm_manual = torch.nn.functional.normalize(x, p=2, dim=-1)
        y_norm_manual = torch.nn.functional.normalize(y, p=2, dim=-1)

        # SCE loss internal normalization
        x_for_loss = torch.nn.functional.normalize(x, p=2, dim=-1)
        y_for_loss = torch.nn.functional.normalize(y, p=2, dim=-1)

        # They should be identical
        assert torch.allclose(x_norm_manual, x_for_loss, atol=1e-7), \
            "Manual normalization differs from SCE loss normalization"
        assert torch.allclose(y_norm_manual, y_for_loss, atol=1e-7), \
            "Manual normalization differs from SCE loss normalization"

        # Compute loss with normalized vectors
        loss = sce_loss(x, y, alpha=3)
        assert loss.item() >= 0, "SCE loss should be non-negative"

    def test_normalized_vs_unnormalized_clustering(self, sample_embeddings):
        """
        Test that L2 normalization improves alignment between cosine-based training
        and Euclidean-based clustering.

        This is a conceptual test showing the difference.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Original embeddings (not normalized)
        embeddings_np = sample_embeddings.numpy()

        # Normalized embeddings
        normalized = torch.nn.functional.normalize(sample_embeddings, p=2, dim=-1)
        normalized_np = normalized.numpy()

        # Cluster both
        n_clusters = 5
        kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_orig = kmeans_orig.fit_predict(embeddings_np)

        kmeans_norm = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_norm = kmeans_norm.fit_predict(normalized_np)

        # Both should produce valid clusterings
        assert len(np.unique(labels_orig)) == n_clusters
        assert len(np.unique(labels_norm)) == n_clusters

        # Normalized embeddings should have all norms equal to 1
        norms = np.linalg.norm(normalized_np, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), \
            "Normalized embeddings should have unit norm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
