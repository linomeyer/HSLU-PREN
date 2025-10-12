"""
Jigsaw Puzzle Solver - Main Entry Point
"""

from edge_extraction import process_all_pieces
from edge_matching import match_puzzle_pieces
from piece_segmentation import PieceSegmenter, PieceVisualizer


def main():
    # Path to your puzzle image
    image_path = "img\\puzzle_image.png"

    print("Starting Jigsaw Puzzle Piece Analysis...")
    print("=" * 60)

    try:
        # ====================================================================
        # Step 1: Segment pieces
        # ====================================================================
        print("\n[Step 1] Segmenting puzzle pieces...")
        print("-" * 60)

        segmenter = PieceSegmenter(image_path)
        pieces = segmenter.segment_pieces(min_area=500, max_area=None)

        if not pieces:
            print("No pieces found. Try adjusting the min_area parameter.")
            return

        print(f"\n✓ Found {len(pieces)} pieces")

        # Display segmentation statistics
        stats = segmenter.get_piece_statistics()
        print(f"  Average area: {stats['avg_area']:.2f} pixels")
        print(f"  Area range: {stats['min_area']:.2f} - {stats['max_area']:.2f}")

        # Visualize segmentation using PieceVisualizer
        print("\nGenerating segmentation visualizations...")
        piece_visualizer = PieceVisualizer(segmenter.image_rgb, pieces)
        piece_visualizer.visualize_segmentation(show_labels=True, show_contours=True)
        piece_visualizer.visualize_all_pieces_grid(cols=4)

        # ====================================================================
        # Step 2: Extract and classify edges
        # ====================================================================
        print("\n[Step 2] Extracting and classifying edges...")
        print("-" * 60)

        extractors = process_all_pieces(pieces, visualize=True)

        # ====================================================================
        # Step 3: Match edges between pieces
        # ====================================================================
        print("\n[Step 3] Matching puzzle piece edges...")
        print("-" * 60)

        matcher = match_puzzle_pieces(extractors, shape_weight=0.7, color_weight=0.3, visualize=True)

        # ====================================================================
        # Step 4: Summary and Analysis
        # ====================================================================
        print("\n" + "=" * 60)
        print("PUZZLE ANALYSIS SUMMARY")
        print("=" * 60)

        corner_pieces = []
        border_pieces = []
        interior_pieces = []

        for extractor in extractors:
            summary = extractor.get_edge_summary()
            if summary:
                if summary['is_corner_piece']:
                    corner_pieces.append(summary['piece_id'])
                elif summary['is_border_piece']:
                    border_pieces.append(summary['piece_id'])
                else:
                    interior_pieces.append(summary['piece_id'])

        print(f"\nTotal pieces: {len(pieces)}")
        print(f"  Corner pieces: {len(corner_pieces)} - {corner_pieces}")
        print(f"  Border pieces: {len(border_pieces)} - {border_pieces}")
        print(f"  Interior pieces: {len(interior_pieces)} - {interior_pieces}")

        # Detailed breakdown
        print("\nDetailed Edge Classification:")
        print("-" * 60)
        for extractor in extractors:
            summary = extractor.get_edge_summary()
            if summary:
                piece_type = "CORNER" if summary['is_corner_piece'] else \
                    "BORDER" if summary['is_border_piece'] else "INTERIOR"
                print(f"\nPiece {summary['piece_id']} ({piece_type}):")
                for edge_info in summary['edges']:
                    print(f"  {edge_info['direction']:7s}: {edge_info['type']:7s} "
                          f"({edge_info['num_points']} points, "
                          f"curvature: {edge_info['curvature']:.4f})")

        print("\n" + "=" * 60)
        print("✓ Analysis complete!")
        print("=" * 60)

    except FileNotFoundError:
        print(f"\nError: Could not find image at '{image_path}'")
        print("Please update the image_path variable with a valid image file.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
