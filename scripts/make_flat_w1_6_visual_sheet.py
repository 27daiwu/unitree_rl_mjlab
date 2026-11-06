"""Make synchronized Control/Experiment frame sheets from W1.6 videos."""

from pathlib import Path
import argparse

import imageio.v2 as imageio
from PIL import Image, ImageDraw


TIMES = (8.0, 8.1, 8.2, 8.3, 8.4, 8.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/flat_w1_6_toe_clearance_20261030/paired_visual"))
    args = parser.parse_args()
    root = args.root
    for scenario in ("backward", "left", "right", "forward", "turn_left", "turn_right"):
        paths = [root / f"{label}_{scenario}.mp4" for label in ("control", "experiment")]
        if not all(path.exists() for path in paths):
            continue
        sheet = Image.new("RGB", (480 * len(TIMES), 390 * 2), "white")
        draw = ImageDraw.Draw(sheet)
        for row, path in enumerate(paths):
            reader = imageio.get_reader(path)
            try:
                for col, time in enumerate(TIMES):
                    frame = Image.fromarray(reader.get_data(round(time * 50))).convert("RGB")
                    if root.name != "paired_visual_close":
                        frame = frame.crop((160, 300, 640, 660))
                    else:
                        frame.thumbnail((480, 360))
                    sheet.paste(frame, (480 * col, 390 * row + 30))
                    draw.text((480 * col + 8, 390 * row + 8),
                              f"{path.stem}  {time:.1f}s", fill="black")
            finally:
                reader.close()
        sheet.save(root / f"sheet_{scenario}.jpg", quality=90)


if __name__ == "__main__":
    main()
