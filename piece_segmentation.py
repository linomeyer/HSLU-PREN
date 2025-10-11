from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import Mat


class PuzzlePiece:
    """Represents a single puzzle piece with its properties."""

    def __init__(self, contour, mask, bbox, image_roi):
        self.contour = contour
        self.mask = mask
        self.bbox = bbox  # (x, y, w, h)
        self.image_roi = image_roi  # The piece image with transparent background
        self.center = self._calculate_center()

    def _calculate_center(self):
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        return 0, 0


class PieceSegmenter:
    """Segments puzzle pieces from an image."""

    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.pieces: List[PuzzlePiece] = []

    def segment_pieces(self, min_area: int = 1000, max_area: int = None) -> List[PuzzlePiece]:
        """
        Segment puzzle pieces from the image.

        Args:
            min_area: Minimum contour area to consider as a piece
            max_area: Maximum contour area (default: 1/4 of image area)

        Returns:
            List of PuzzlePiece objects
        """
        if max_area is None:
            max_area = (self.image.shape[0] * self.image.shape[1]) // 4

        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 3: Threshold to separate pieces from background
        # method 1
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Alternative: Adaptive threshold (good for uneven lighting)
        # method 2
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

        # default method = 1
        thresh = thresh1

        # Step 4: Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Step 5: Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 6: Filter and create PuzzlePiece objects
        self.pieces = []
        for contour in contours:
            puzzle_piece = self._create_puzzle_piece(contour, gray, min_area, max_area)
            if puzzle_piece is not None:
                self.pieces.append(puzzle_piece)

        print(f"Found {len(self.pieces)} puzzle pieces")
        return self.pieces

    def _create_puzzle_piece(self, contour: Mat, gray_scale: Mat, min_area: int, max_area: int) -> PuzzlePiece | None:
        area = cv2.contourArea(contour)

        # Filter by area
        if area < min_area or area > max_area:
            return None

        # Create mask for this piece
        mask = np.zeros(gray_scale.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Extract piece image with mask
        piece_mask_roi = mask[y:y + h, x:x + w]
        piece_image_roi = self.image_rgb[y:y + h, x:x + w].copy()

        # Create RGBA image (with alpha channel for transparency)
        piece_rgba = cv2.cvtColor(piece_image_roi, cv2.COLOR_RGB2RGBA)
        piece_rgba[:, :, 3] = piece_mask_roi

        # Create PuzzlePiece object
        return PuzzlePiece(contour, mask, (x, y, w, h), piece_rgba)

    def get_piece_statistics(self) -> Dict:
        """Get statistics about the segmented pieces."""
        if not self.pieces:
            return {}

        areas = [cv2.contourArea(piece.contour) for piece in self.pieces]
        perimeters = [cv2.arcLength(piece.contour, True) for piece in self.pieces]

        stats = {
            'num_pieces': len(self.pieces),
            'avg_area': np.mean(areas),
            'std_area': np.std(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'avg_perimeter': np.mean(perimeters),
        }

        return stats


class PieceVisualizer:
    def __init__(self, image_rgb: Mat, pieces: List[PuzzlePiece]):
        self.image_rgb = image_rgb
        self.pieces = pieces

    def visualize_segmentation(self, show_labels: bool = True, show_contours: bool = True):
        """
        Visualize the segmented pieces in multiple views.

        Args:
            show_labels: Whether to show piece numbers
            show_contours: Whether to show contour outlines
        """
        if not self.pieces:
            print("No pieces found. Run segment_pieces() first.")
            return

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))

        # 1. Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(self.image_rgb)
        ax1.set_title("Original Image")
        ax1.axis('off')

        # 2. Image with contours
        ax2 = plt.subplot(2, 3, 2)
        contour_image = self.image_rgb.copy()
        for i, piece in enumerate(self.pieces):
            color = np.random.randint(0, 255, 3).tolist()
            cv2.drawContours(contour_image, [piece.contour], -1, color, 3)

            if show_labels:
                cv2.putText(contour_image, str(i), piece.center,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ax2.imshow(contour_image)
        ax2.set_title(f"Detected Pieces ({len(self.pieces)})")
        ax2.axis('off')

        # 3. Colored masks
        ax3 = plt.subplot(2, 3, 3)
        colored_masks = np.zeros_like(self.image_rgb)
        for piece in self.pieces:
            color = np.random.randint(50, 255, 3).tolist()
            colored_masks[piece.mask > 0] = color

        ax3.imshow(colored_masks)
        ax3.set_title("Colored Masks")
        ax3.axis('off')

        # 4-6. Individual pieces (show first 3)
        for i in range(min(3, len(self.pieces))):
            ax = plt.subplot(2, 3, 4 + i)
            ax.imshow(self.pieces[i].image_roi)
            ax.set_title(f"Piece {i}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_all_pieces_grid(self, cols: int = 4):
        """
        Visualize all pieces in a grid layout.

        Args:
            cols: Number of columns in the grid
        """
        if not self.pieces:
            print("No pieces found. Run segment_pieces() first.")
            return

        n_pieces = len(self.pieces)
        rows = (n_pieces + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if n_pieces > 1 else [axes]

        for i, piece in enumerate(self.pieces):
            axes[i].imshow(piece.image_roi)
            axes[i].set_title(f"Piece {i}\nArea: {cv2.contourArea(piece.contour):.0f}")
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_pieces, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
