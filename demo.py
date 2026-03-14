from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ccrl import CCRLConfig, run_ccrl_diag


def guess_data_path() -> Path | None:
    candidates = [
        ROOT / "data" / "fault_dataset.pkl",
        ROOT / "data" / "data_merge.pkl",
        ROOT.parent / "data" / "fault_dataset.pkl",
        ROOT.parent / "data" / "data_merge.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    data_path = guess_data_path()
    if data_path is None or not data_path.exists():
        print(
            "Input pickle not found. Put fault_dataset.pkl or data_merge.pkl under ./data, or edit demo.py to point to your file.",
            file=sys.stderr,
        )
        return 1

    config = CCRLConfig()
    config.data.pretrain_label = "normal"
    result = run_ccrl_diag(
        data_path=data_path,
        config=config,
        repeats=1,
        seed=2024,
        log_dir=ROOT / "logs",
    )
    print(json.dumps({"data": str(data_path), "mean_f1": result.mean_f1, "std_f1": result.std_f1}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
