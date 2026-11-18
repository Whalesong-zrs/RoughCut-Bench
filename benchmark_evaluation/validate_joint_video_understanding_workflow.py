"""Validate repeated joint_video_understanding_workflow outputs.

This is a workflow-specific wrapper around validate_visual_workflow.py. It keeps
all metric and coverage logic shared, but writes joint-understanding-specific
markdown filenames and titles by default.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_evaluation.validate_visual_workflow import main


def add_default(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    add_default("--understanding-output-name", "joint_video_understanding.md")
    add_default("--cut-planning-output-name", "joint_understanding_cut_planning.md")
    add_default("--understanding-title", "Joint video-understanding validation")
    add_default("--cut-planning-title", "Joint-understanding Cut-Planning Agent validation")
    main()
