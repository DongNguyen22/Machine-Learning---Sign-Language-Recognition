"""Điểm vào kiểm tra tầng cấu hình: nạp YAML và ghi một dòng log.

Chạy từ thư mục gốc dự án:

    python scripts/check_config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logger import get_logger, setup_logger  # noqa: E402


def main() -> None:
    """Nạp cấu hình và xác nhận logger ghi được ra file."""
    config = load_config()
    log_path = config.paths.logs_dir / "asl.log"
    setup_logger(log_file=log_path)
    log = get_logger("asl.config")
    log.info(
        "Config OK | project=%s | classes=%d | features=%d | seed=%d",
        config.project_name,
        config.mlp.num_classes,
        config.mlp.input_dim,
        config.split.random_seed,
    )
    for handler in log.handlers:
        handler.flush()
    print(
        "Config OK | "
        f"project={config.project_name} | "
        f"classes={config.mlp.num_classes} | "
        f"features={config.mlp.input_dim} | "
        f"seed={config.split.random_seed}"
    )
    print(f"Log: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"Thiếu file: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
