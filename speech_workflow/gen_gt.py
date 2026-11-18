#!/usr/bin/env python3
"""Generate plain-text references from `*_gt.txt` by stripping annotation tags."""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_workflow.runtime_paths import SPEECH_DATA_ROOT


def process_text(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(SPEECH_DATA_ROOT), help="Directory containing *_gt.txt files")
    args = parser.parse_args()
    dir_path = Path(args.dir)

    for gt_path in sorted(dir_path.glob("*_gt.txt")):
        with gt_path.open("r", encoding="utf-8") as f:
            original = f.read()

        test = process_text(original)
        test_path = gt_path.with_name(f"{gt_path.stem[:-3]}.txt")

        with test_path.open("w", encoding="utf-8") as f:
            f.write(test)

        print(f"Generated: {test_path.name}")


if __name__ == "__main__":
    main()
