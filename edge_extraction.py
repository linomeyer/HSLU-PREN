from enum import Enum
from typing import List, Tuple, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np


class EdgeType(Enum):
    """Types of puzzle piece edges."""
    FLAT = "flat"  # Straight edge (border piece)
    TAB = "tab"  # Protruding edge (male)
    BLANK = "blank"  # Indented edge (female)
    UNKNOWN = "unknown"


class PuzzleEdge:
    """Represents a single edge of a puzzle piece."""

    def __init__(self, points: np.ndarray, edge_type: EdgeType, edge_index: int):
        self.points = points  # Array of (x, y) coordinates along the edge
        self.edge_type = edge_type
        self.edge_index = edge_index  # 0=top, 1=right, 2=bottom, 3=left
        self.curvature = None
        self.color_profile = None

    def get_direction_name(self) -> str:
        """Get human-readable direction name."""
        directions = ["Top", "Right", "Bottom", "Left"]
        return directions[self.edge_index]


class EdgeExtractor:
    """Extracts and classifies edges from puzzle pieces."""

    def __init__(self, piece, piece_id: int = 0):
        """
        Initialize edge extractor for a puzzle piece.

        Args:
            piece: PuzzlePiece object from piece_segmentation
            piece_id: Identifier for the piece
        """
        self.piece = piece
        self.piece_id = piece_id
        self.edges: List[PuzzleEdge] = []
        self.corners: List[Tuple[int, int]] = []

    def extract_edges(self, num_corners: int = 4) -> List[PuzzleEdge]:
        """
        Extract and classify all edges of the puzzle piece.

        Args:
            num_corners: Expected number of corners (default: 4)

        Returns:
            List of PuzzleEdge objects
        """
        contour = self.piece.contour

        # Step 1: Find corners
        self.corners = self._find_corners(contour, num_corners)

        if len(self.corners) < num_corners:
            print(f"Warning: Found only {len(self.corners)} corners for piece {self.piece_id}")
            return []

        # Step 2: Split contour into edges between corners
        edge_segments = self._split_contour_by_corners(contour, self.corners)

        # Step 3: Classify each edge
        self.edges = []
        for i, segment in enumerate(edge_segments):
            edge_type = self._classify_edge(segment)
            edge = PuzzleEdge(segment, edge_type, i)
            edge.curvature = self._calculate_curvature(segment)
            self.edges.append(edge)

        return self.edges

    def _find_corners(self, contour: np.ndarray, num_corners: int = 4) -> List[Tuple[int, int]]:
        """
        Find corner points of the puzzle piece using a simple bounding box approach.
        Since puzzle pieces are quadrilateral, we can use the oriented bounding box.

        Args:
            contour: Contour points
            num_corners: Expected number of corners

        Returns:
            List of corner coordinates
        """
        # Use minimum area rectangle (oriented bounding box)
        # This works great for quadrilateral puzzle pieces
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)  # Convert to integers

        # Convert to list of tuples
        corners = [tuple(point) for point in box]

        # Order the corners properly
        return self._order_corners(corners)

    def _order_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Order corners in consistent order: [top-left, top-right, bottom-right, bottom-left]

        Args:
            corners: List of 4 corner points

        Returns:
            Ordered list of corners
        """
        if len(corners) != 4:
            return corners

        # Convert to numpy array
        corners_array = np.array(corners, dtype=np.float32)

        # Find center point
        center = np.mean(corners_array, axis=0)

        # Calculate angle from center to each corner
        angles = np.arctan2(corners_array[:, 1] - center[1],
                            corners_array[:, 0] - center[0])

        # Sort by angle (counterclockwise from right)
        sorted_indices = np.argsort(angles)
        sorted_corners = corners_array[sorted_indices]

        # Find top-left (smallest x+y sum)
        sums = sorted_corners[:, 0] + sorted_corners[:, 1]
        top_left_idx = np.argmin(sums)

        # Rotate array so top-left is first, then reverse for clockwise
        # Going clockwise: top-left -> top-right -> bottom-right -> bottom-left
        reordered = np.roll(sorted_corners, -top_left_idx, axis=0)

        # Check if we need to reverse (make sure we go clockwise)
        # If second point is to the left of first point, we're going counterclockwise
        if reordered[1][0] < reordered[0][0]:
            reordered = np.vstack([reordered[0], reordered[:0:-1]])

        return [tuple(corner) for corner in reordered]

    def _split_contour_by_corners(self, contour: np.ndarray, corners: List[Tuple[int, int]]) -> List[np.ndarray]:
        """
        Split contour into 4 edge segments between corner points.
        Returns edges in order: [top, right, bottom, left]

        Args:
            contour: The contour points
            corners: Ordered corner points [top-left, top-right, bottom-right, bottom-left]

        Returns:
            List of 4 edge segments
        """
        contour_points = contour.reshape(-1, 2)

        # Find the contour indices closest to each corner
        corner_indices = []
        for corner in corners:
            distances = np.linalg.norm(contour_points - np.array(corner), axis=1)
            closest_idx = np.argmin(distances)
            corner_indices.append(closest_idx)

        # Ensure indices are in contour order
        corner_indices = sorted(corner_indices)

        # Split contour into 4 segments
        segments = []
        num_corners = len(corner_indices)

        for i in range(num_corners):
            start_idx = corner_indices[i]
            end_idx = corner_indices[(i + 1) % num_corners]

            if end_idx > start_idx:
                segment = contour_points[start_idx:end_idx + 1]
            else:
                # Handle wrap-around at the end of contour
                segment = np.vstack([
                    contour_points[start_idx:],
                    contour_points[:end_idx + 1]
                ])

            segments.append(segment)

        return segments

    def _classify_edge(self, edge_points: np.ndarray) -> EdgeType:
        """
        Classify an edge as FLAT, TAB, or BLANK.

        Args:
            edge_points: Array of points along the edge

        Returns:
            EdgeType classification
        """
        if len(edge_points) < 3:
            return EdgeType.UNKNOWN

        # Calculate the straight line distance between endpoints
        start_point = edge_points[0]
        end_point = edge_points[-1]
        straight_distance = np.linalg.norm(end_point - start_point)

        # Calculate the actual path length along the edge
        path_length = np.sum(np.linalg.norm(np.diff(edge_points, axis=0), axis=1))

        # Calculate deviation ratio
        if straight_distance > 0:
            deviation_ratio = path_length / straight_distance
        else:
            return EdgeType.UNKNOWN

        # Threshold for flat edge (small deviation = straight line)
        FLAT_THRESHOLD = 1.12  # Less than 12% deviation (more strict)

        if deviation_ratio < FLAT_THRESHOLD:
            return EdgeType.FLAT

        # For curved edges, determine if it's a tab or blank
        # by checking if the edge bulges outward or inward

        # Calculate the direction perpendicular to the edge
        edge_vector = end_point - start_point
        edge_length = np.linalg.norm(edge_vector)

        if edge_length == 0:
            return EdgeType.UNKNOWN

        edge_direction = edge_vector / edge_length
        perpendicular = np.array([-edge_direction[1], edge_direction[0]])

        # Calculate signed distances for all points
        signed_distances = []
        for point in edge_points:
            # Vector from start to this point
            point_vector = point - start_point

            # Project onto perpendicular direction
            projection = np.dot(point_vector, perpendicular)
            signed_distances.append(projection)

        signed_distances = np.array(signed_distances)

        # Use the median of the absolute values to determine direction
        # This is more robust than just finding the maximum
        positive_distances = signed_distances[signed_distances > 0]
        negative_distances = signed_distances[signed_distances < 0]

        # Calculate the "mass" on each side
        positive_mass = np.sum(positive_distances) if len(positive_distances) > 0 else 0
        negative_mass = np.sum(np.abs(negative_distances)) if len(negative_distances) > 0 else 0

        # Threshold for distinguishing tab from blank
        TAB_BLANK_THRESHOLD = straight_distance * 0.08  # 8% of edge length

        max_deviation = max(abs(positive_mass), abs(negative_mass)) / len(edge_points)

        if max_deviation < TAB_BLANK_THRESHOLD:
            return EdgeType.FLAT
        elif positive_mass > negative_mass:
            return EdgeType.TAB
        else:
            return EdgeType.BLANK

    def _calculate_curvature(self, edge_points: np.ndarray) -> float:
        """Calculate average curvature along the edge."""
        if len(edge_points) < 3:
            return 0.0

        curvatures = []
        for i in range(1, len(edge_points) - 1):
            p1 = edge_points[i - 1]
            p2 = edge_points[i]
            p3 = edge_points[i + 1]

            # Calculate curvature using three points
            v1 = p2 - p1
            v2 = p3 - p2

            angle = self._angle_between_vectors(v1, v2)
            curvatures.append(abs(angle))

        return np.mean(curvatures) if curvatures else 0.0

    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in radians."""
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return 0.0

        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.arccos(cos_angle)

    def visualize_edges(self, show_corners: bool = True, show_labels: bool = True):
        """
        Visualize the extracted edges with their classifications.
        
        Args:
            show_corners: Whether to mark corner points
            show_labels: Whether to show edge type labels
        """
        if not self.edges:
            print("No edges extracted. Run extract_edges() first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Get bounding box for better visualization
        x, y, w, h = self.piece.bbox
        piece_image = self.piece.image_roi.copy()

        # Draw on a copy of the original image
        viz_image = cv2.cvtColor(piece_image[:, :, :3], cv2.COLOR_RGB2BGR)

        # Color scheme for edge types
        edge_colors = {
            EdgeType.FLAT: (0, 255, 0),  # Green
            EdgeType.TAB: (255, 0, 0),  # Blue
            EdgeType.BLANK: (0, 0, 255),  # Red
            EdgeType.UNKNOWN: (128, 128, 128)  # Gray
        }

        # Draw each edge with its classification color
        for edge in self.edges:
            color = edge_colors[edge.edge_type]

            # Adjust points to local coordinates (relative to bounding box)
            local_points = edge.points - np.array([x, y])
            local_points = local_points.astype(np.int32)

            # Draw edge
            for i in range(len(local_points) - 1):
                cv2.line(viz_image, tuple(local_points[i]), tuple(local_points[i + 1]), color, 3)

            # Add label
            if show_labels:
                mid_point = local_points[len(local_points) // 2]
                label = f"{edge.get_direction_name()}: {edge.edge_type.value}"
                cv2.putText(viz_image, label, tuple(mid_point),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw corners
        if show_corners and self.corners:
            for corner in self.corners:
                local_corner = (int(corner[0]) - x, int(corner[1]) - y)
                cv2.circle(viz_image, local_corner, 8, (255, 255, 0), -1)
                cv2.circle(viz_image, local_corner, 10, (0, 0, 0), 2)

        # Display original and annotated
        axes[0].imshow(piece_image)
        axes[0].set_title(f"Piece {self.piece_id} - Original")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Piece {self.piece_id} - Edge Classification")
        axes[1].axis('off')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=np.array(edge_colors[EdgeType.FLAT]) / 255, lw=4, label='Flat (Border)'),
            plt.Line2D([0], [0], color=np.array(edge_colors[EdgeType.TAB]) / 255, lw=4, label='Tab (Outward)'),
            plt.Line2D([0], [0], color=np.array(edge_colors[EdgeType.BLANK]) / 255, lw=4, label='Blank (Inward)'),
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()

    def get_edge_summary(self) -> Dict:
        """Get summary statistics about the edges."""
        if not self.edges:
            return {}

        edge_type_counts = {
            EdgeType.FLAT: 0,
            EdgeType.TAB: 0,
            EdgeType.BLANK: 0,
            EdgeType.UNKNOWN: 0
        }

        for edge in self.edges:
            edge_type_counts[edge.edge_type] += 1

        summary = {
            'piece_id': self.piece_id,
            'num_edges': len(self.edges),
            'flat_edges': edge_type_counts[EdgeType.FLAT],
            'tab_edges': edge_type_counts[EdgeType.TAB],
            'blank_edges': edge_type_counts[EdgeType.BLANK],
            'unknown_edges': edge_type_counts[EdgeType.UNKNOWN],
            'is_corner_piece': edge_type_counts[EdgeType.FLAT] == 2,
            'is_border_piece': edge_type_counts[EdgeType.FLAT] >= 1,
            'edges': [
                {
                    'direction': edge.get_direction_name(),
                    'type': edge.edge_type.value,
                    'num_points': len(edge.points),
                    'curvature': edge.curvature
                }
                for edge in self.edges
            ]
        }

        return summary


def process_all_pieces(pieces: List, visualize: bool = True) -> List[EdgeExtractor]:
    """
    Process all pieces to extract and classify edges.
    
    Args:
        pieces: List of PuzzlePiece objects
        visualize: Whether to show visualizations
        
    Returns:
        List of EdgeExtractor objects
    """
    extractors = []

    for i, piece in enumerate(pieces):
        print(f"\nProcessing piece {i}...")
        extractor = EdgeExtractor(piece, piece_id=i)
        edges = extractor.extract_edges()

        if edges:
            summary = extractor.get_edge_summary()
            print(f"  Edges: {summary['flat_edges']} flat, {summary['tab_edges']} tabs, {summary['blank_edges']} blanks")

            if summary['is_corner_piece']:
                print(f"  -> This is a CORNER piece!")
            elif summary['is_border_piece']:
                print(f"  -> This is a BORDER piece!")

            if visualize and i < 5:  # Show first 5 pieces
                extractor.visualize_edges()

        extractors.append(extractor)

    return extractors
