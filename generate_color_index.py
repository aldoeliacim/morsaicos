#!/usr/bin/env python3
"""
Script to generate the color index for mosaic generation.
"""

import os
import sys
from pathlib import Path

from index_generator import IndexGenerator

def main() -> bool:
    print("Generating perceptual color signatures (CIEDE2000) for mosaic images...")

    if not os.path.exists("img"):
        print("img directory not found!")
        return False

    img_count = len([
        f for f in os.listdir("img")
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"))
    ])
    print(f"Found {img_count} images to process")

    try:
        generator = IndexGenerator(
            "img",
            output_directory=".",
            remote_base_url="https://img.get.aldo.pw"
        )

        print("Processing images...")
        color_index = generator.generate_and_save(
            use_multiprocessing=True,
            save_format="both"
        )

        if color_index:
            print(f"Generated color index with {len(color_index)} entries")
            print("Saved to color_index.[pkl|json|txt] in project root")

            # Keep a mirror inside img/ for local inspection or syncing
            import shutil
            target_dir = Path("img")
            if target_dir.exists():
                for suffix in ("pkl", "json", "txt"):
                    src = Path(f"color_index.{suffix}")
                    if src.exists():
                        dst = target_dir / src.name
                        shutil.copy2(src, dst)
                print("Mirrored index files into img/ directory")
            return True

        print("Failed to generate color index")
        return False

    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"Error generating color index: {exc}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
