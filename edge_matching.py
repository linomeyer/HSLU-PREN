"""
Edge Matching Module - Matches puzzle piece edges based on shape and color similarity
"""

from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import directed_hausdorff

from edge_extraction import EdgeExtractor, EdgeType, PuzzleEdge


class EdgeMatch:
    """Represents a potential match between two edges."""

    def __init__(self, piece1_id: int, edge1_idx: int, piece2_id: int, edge2_idx: int, score: float):
        self.piece1_id = piece1_id
        self.edge1_idx = edge1_idx
        self.piece2_id = piece2_id
        self.edge2_idx = edge2_idx
        self.score = score  # Lower is better (0 = perfect match)

    def __repr__(self):
        return f"EdgeMatch(P{self.piece1_id}:E{self.edge1_idx} <-> P{self.piece2_id}:E{self.edge2_idx}, score={self.score:.3f})"


class EdgeMatcher:
    """Matches edges between puzzle pieces."""

    def __init__(self, extractors: List[EdgeExtractor]):
        """
        Initialize edge matcher with extracted edges.

        Args:
            extractors: List of EdgeExtractor objects with extracted edges
        """
        self.extractors = extractors
        self.matches: List[EdgeMatch] = []

    def find_all_matches(self, shape_weight: float = 0.7, color_weight: float = 0.3,
                         max_matches_per_edge: int = 3) -> List[EdgeMatch]:
        """
        Find all potential edge matches between pieces.

        Args:
            shape_weight: Weight for shape similarity (0-1)
            color_weight: Weight for color similarity (0-1)
            max_matches_per_edge: Maximum number of matches to keep per edge

        Returns:
            List of EdgeMatch objects sorted by score
        """
        print("\nFinding edge matches...")
        self.matches = []

        # Iterate through all pairs of pieces
        for i, ext1 in enumerate(self.extractors):
            for j, ext2 in enumerate(self.extractors):
                if i >= j:  # Skip same piece and avoid duplicate pairs
                    continue

                # Try to match each edge of piece1 with each edge of piece2
                for edge1_idx, edge1 in enumerate(ext1.edges):
                    for edge2_idx, edge2 in enumerate(ext2.edges):
                        # Check if edges are compatible (tab with blank, or both flat)
                        if self._are_edges_compatible(edge1, edge2):
                            # Calculate match score
                            score = self._calculate_match_score(
                                ext1, edge1, edge1_idx,
                                ext2, edge2, edge2_idx,
                                shape_weight, color_weight
                            )

                            match = EdgeMatch(i, edge1_idx, j, edge2_idx, score)
                            self.matches.append(match)

        # Sort by score (lower is better)
        self.matches.sort(key=lambda x: x.score)

        # Filter to keep only top matches per edge
        filtered_matches = self._filter_top_matches(max_matches_per_edge)

        print(f"Found {len(filtered_matches)} potential matches")
        return filtered_matches

    def _are_edges_compatible(self, edge1: PuzzleEdge, edge2: PuzzleEdge) -> bool:
        """
        Check if two edges are compatible for matching.
        TAB matches with BLANK, FLAT matches with FLAT.
        """
        # Both flat (border pieces)
        if edge1.edge_type == EdgeType.FLAT and edge2.edge_type == EdgeType.FLAT:
            return True

        # Tab matches with blank
        if (edge1.edge_type == EdgeType.TAB and edge2.edge_type == EdgeType.BLANK) or \
                (edge1.edge_type == EdgeType.BLANK and edge2.edge_type == EdgeType.TAB):
            return True

        return False

    def _calculate_match_score(self, ext1: EdgeExtractor, edge1: PuzzleEdge, edge1_idx: int,
                               ext2: EdgeExtractor, edge2: PuzzleEdge, edge2_idx: int,
                               shape_weight: float, color_weight: float) -> float:
        """
        Calculate similarity score between two edges.
        Lower score = better match.
        """
        # Shape similarity
        shape_score = self._calculate_shape_similarity(edge1, edge2)

        # Color similarity along the edge
        color_score = self._calculate_color_similarity(ext1, edge1, ext2, edge2)

        # Combined score
        total_score = shape_weight * shape_score + color_weight * color_score

        return total_score

    def _calculate_shape_similarity(self, edge1: PuzzleEdge, edge2: PuzzleEdge) -> float:
        """
        Calculate shape similarity using Hausdorff distance and shape descriptors.
        """
        # Normalize edges to same length for comparison
        normalized1 = self._normalize_edge(edge1.points)
        normalized2 = self._normalize_edge(edge2.points)

        # For tab-blank matching, we need to flip one edge
        if edge1.edge_type != edge2.edge_type:
            # Flip edge2 (mirror it)
            normalized2 = self._flip_edge(normalized2)

        # Calculate Hausdorff distance (measures shape similarity)
        hausdorff_dist = max(
            directed_hausdorff(normalized1, normalized2)[0],
            directed_hausdorff(normalized2, normalized1)[0]
        )

        # Normalize by edge length
        edge_length = np.linalg.norm(edge1.points[-1] - edge1.points[0])
        if edge_length > 0:
            hausdorff_dist /= edge_length

        return hausdorff_dist

    def _normalize_edge(self, points: np.ndarray, num_samples: int = 100) -> np.ndarray:
        """
        Normalize edge to a standard representation with fixed number of points.
        """
        if len(points) < 2:
            return points

        # Calculate cumulative distance along the edge
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            distances[i] = distances[i - 1] + np.linalg.norm(points[i] - points[i - 1])

        # Normalize distances to [0, 1]
        if distances[-1] > 0:
            distances = distances / distances[-1]

        # Interpolate to get fixed number of samples
        interp_x = interp1d(distances, points[:, 0], kind='linear')
        interp_y = interp1d(distances, points[:, 1], kind='linear')

        new_distances = np.linspace(0, 1, num_samples)
        normalized_points = np.column_stack([interp_x(new_distances), interp_y(new_distances)])

        # Translate to origin (start point at 0,0)
        normalized_points = normalized_points - normalized_points[0]

        # Rotate so end point is horizontal
        end_point = normalized_points[-1]
        angle = np.arctan2(end_point[1], end_point[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normalized_points = normalized_points @ rotation_matrix.T

        # Scale to unit length
        length = np.linalg.norm(normalized_points[-1])
        if length > 0:
            normalized_points = normalized_points / length

        return normalized_points

    def _flip_edge(self, points: np.ndarray) -> np.ndarray:
        """
        Flip edge vertically (for matching tab with blank).
        """
        flipped = points.copy()
        flipped[:, 1] = -flipped[:, 1]
        return flipped

    def _calculate_color_similarity(self, ext1: EdgeExtractor, edge1: PuzzleEdge,
                                    ext2: EdgeExtractor, edge2: PuzzleEdge) -> float:
        """
        Calculate color similarity along the edges.
        """
        # Extract color profiles along edges
        color_profile1 = self._extract_color_profile(ext1.piece, edge1.points)
        color_profile2 = self._extract_color_profile(ext2.piece, edge2.points)

        if color_profile1 is None or color_profile2 is None:
            return 0.5  # Neutral score if color extraction fails

        # Resample to same length
        len1, len2 = len(color_profile1), len(color_profile2)
        if len1 > len2:
            indices = np.linspace(0, len1 - 1, len2).astype(int)
            color_profile1 = color_profile1[indices]
        elif len2 > len1:
            indices = np.linspace(0, len2 - 1, len1).astype(int)
            color_profile2 = color_profile2[indices]

        # Calculate color difference (normalized)
        color_diff = np.mean(np.linalg.norm(color_profile1 - color_profile2, axis=1))

        # Normalize to [0, 1] (assuming max RGB difference is sqrt(3*255^2))
        max_diff = np.sqrt(3 * 255 ** 2)
        normalized_diff = color_diff / max_diff

        return normalized_diff

    def _extract_color_profile(self, piece, edge_points: np.ndarray,
                               sample_width: int = 5) -> Optional[np.ndarray]:
        """
        Extract color profile along an edge.

        Args:
            piece: PuzzlePiece object
            edge_points: Points along the edge
            sample_width: Width of sampling perpendicular to edge

        Returns:
            Array of RGB colors along the edge
        """
        try:
            x, y, w, h = piece.bbox
            image_roi = piece.image_roi[:, :, :3]  # RGB only

            colors = []
            for point in edge_points:
                # Convert to local coordinates
                local_x = int(point[0] - x)
                local_y = int(point[1] - y)

                # Check bounds
                if 0 <= local_x < w and 0 <= local_y < h:
                    # Sample color at this point
                    color = image_roi[local_y, local_x]
                    colors.append(color)

            if len(colors) == 0:
                return None

            return np.array(colors, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Could not extract color profile: {e}")
            return None

    def _filter_top_matches(self, max_matches_per_edge: int) -> List[EdgeMatch]:
        """
        Filter to keep only top N matches for each edge.
        """
        # Group matches by edge
        edge_matches = {}

        for match in self.matches:
            # Create unique keys for each edge
            key1 = (match.piece1_id, match.edge1_idx)
            key2 = (match.piece2_id, match.edge2_idx)

            if key1 not in edge_matches:
                edge_matches[key1] = []
            if key2 not in edge_matches:
                edge_matches[key2] = []

            edge_matches[key1].append(match)
            edge_matches[key2].append(match)

        # Keep only top matches for each edge
        valid_matches = set()
        for key, matches in edge_matches.items():
            matches.sort(key=lambda x: x.score)
            for match in matches[:max_matches_per_edge]:
                valid_matches.add(match)

        # Convert back to list and sort
        filtered = list(valid_matches)
        filtered.sort(key=lambda x: x.score)

        return filtered

    def visualize_matches(self, top_n: int = 10):
        """
        Visualize the top N edge matches.

        Args:
            top_n: Number of top matches to visualize
        """
        if not self.matches:
            print("No matches found. Run find_all_matches() first.")
            return

        n_matches = min(top_n, len(self.matches))
        cols = 2
        rows = (n_matches + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if n_matches > 1 else [axes]

        for i, match in enumerate(self.matches[:n_matches]):
            ext1 = self.extractors[match.piece1_id]
            ext2 = self.extractors[match.piece2_id]

            edge1 = ext1.edges[match.edge1_idx]
            edge2 = ext2.edges[match.edge2_idx]

            # Create visualization
            ax = axes[i]

            # Draw both edges
            self._draw_edge_pair(ax, ext1, edge1, ext2, edge2, match.score)

            ax.set_title(f"Match #{i + 1}: P{match.piece1_id}:E{match.edge1_idx} ↔ "
                         f"P{match.piece2_id}:E{match.edge2_idx}\nScore: {match.score:.3f}")
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_matches, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def _draw_edge_pair(self, ax, ext1: EdgeExtractor, edge1: PuzzleEdge,
                        ext2: EdgeExtractor, edge2: PuzzleEdge, score: float):
        """Draw a pair of matched edges for visualization."""
        # Normalize edges for visualization
        norm1 = self._normalize_edge(edge1.points)
        norm2 = self._normalize_edge(edge2.points)

        # Flip edge2 if it's complementary
        if edge1.edge_type != edge2.edge_type:
            norm2 = self._flip_edge(norm2)

        # Offset edge2 for visualization
        norm2[:, 1] = norm2[:, 1] - 0.3

        # Plot edges
        ax.plot(norm1[:, 0], norm1[:, 1], 'b-', linewidth=2, label=f'P{ext1.piece_id} ({edge1.edge_type.value})')
        ax.plot(norm2[:, 0], norm2[:, 1], 'r-', linewidth=2, label=f'P{ext2.piece_id} ({edge2.edge_type.value})')

        # Mark start and end points
        ax.plot(norm1[0, 0], norm1[0, 1], 'go', markersize=8)
        ax.plot(norm1[-1, 0], norm1[-1, 1], 'gs', markersize=8)
        ax.plot(norm2[0, 0], norm2[0, 1], 'ro', markersize=8)
        ax.plot(norm2[-1, 0], norm2[-1, 1], 'rs', markersize=8)

        ax.legend()

    def get_match_statistics(self) -> Dict:
        """Get statistics about the matches."""
        if not self.matches:
            return {}

        scores = [m.score for m in self.matches]

        stats = {
            'num_matches': len(self.matches),
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
        }

        return stats


def match_puzzle_pieces(extractors: List[EdgeExtractor],
                        shape_weight: float = 0.7,
                        color_weight: float = 0.3,
                        visualize: bool = True) -> EdgeMatcher:
    """
    Main function to match puzzle pieces.

    Args:
        extractors: List of EdgeExtractor objects
        shape_weight: Weight for shape similarity
        color_weight: Weight for color similarity
        visualize: Whether to show visualizations

    Returns:
        EdgeMatcher object with found matches
    """
    matcher = EdgeMatcher(extractors)
    matches = matcher.find_all_matches(shape_weight, color_weight)

    if matches:
        stats = matcher.get_match_statistics()
        print(f"\nMatch Statistics:")
        print(f"  Total matches: {stats['num_matches']}")
        print(f"  Average score: {stats['avg_score']:.3f}")
        print(f"  Best score: {stats['min_score']:.3f}")
        print(f"  Worst score: {stats['max_score']:.3f}")

        if visualize:
            matcher.visualize_matches(top_n=10)

    return matcher
