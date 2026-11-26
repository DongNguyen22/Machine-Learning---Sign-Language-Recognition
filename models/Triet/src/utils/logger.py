"""Logger dùng chung: ghi đồng thời ra console và file xoay vòng.

Gọi ``setup_logger`` một lần ở điểm vào chương trình. Logger con
(``asl.training``, ``asl.inference``) kế thừa handler của logger cha ``asl``
nên không bị ghi đúp.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.config_loader import project_root_from

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_HANDLER_MARK = "_asl_managed"
_DEFAULT_LOGGER_NAME = "asl"


def setup_logger(
    name: str = _DEFAULT_LOGGER_NAME,
    log_file: Path | str | None = None,
    level: int | str = logging.INFO,
    *,
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """Gắn handler console (stderr) và file cho ``name``.

    Gọi lại sẽ thay các handler do hàm này tạo, không nhân đôi dòng log.

    Args:
        name: Tên logger. Nên là ``asl`` hoặc một hậu tố của nó.
        log_file: File đích. ``None`` thì ghi vào ``reports/logs/asl.log``
            trong thư mục gốc dự án.
        level: Mức log (số hoặc tên như ``"INFO"``).
        max_bytes: Ngưỡng xoay file.
        backup_count: Số file cũ được giữ.

    Returns:
        Logger đã cấu hình, ``propagate=False`` để không đẩy tiếp lên root.

    Raises:
        FileNotFoundError: Không xác định được thư mục gốc, hoặc không tạo /
            không ghi được file log vì đường dẫn cha không hợp lệ.
        ValueError: ``level`` không phải mức log hợp lệ, hoặc tham số xoay file
            không dương.
    """
    if max_bytes < 1 or backup_count < 1:
        raise ValueError("max_bytes và backup_count phải >= 1.")
    resolved_level = _coerce_level(level)
    path = _resolve_log_file(log_file)
    _ensure_log_parent(path)

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False
    retired = [
        handler
        for handler in logger.handlers
        if getattr(handler, _HANDLER_MARK, False)
    ]
    for handler in retired:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(resolved_level)
    console.setFormatter(formatter)
    setattr(console, _HANDLER_MARK, True)
    logger.addHandler(console)

    try:
        file_handler: logging.Handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FileNotFoundError(f"Không mở được file log {path}: {exc}") from exc
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, _HANDLER_MARK, True)
    logger.addHandler(file_handler)
    return logger


def get_logger(name: str = _DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Lấy logger. Hãy gọi ``setup_logger`` trước ở điểm vào tiến trình.

    Args:
        name: Tên logger, ví dụ ``asl.training``.

    Returns:
        Logger chuẩn của thư viện ``logging``. Nếu chưa setup, logger này
        chưa có handler riêng và sẽ đi theo cấu hình root.
    """
    return logging.getLogger(name)


def _resolve_log_file(log_file: Path | str | None) -> Path:
    if log_file is not None:
        return Path(log_file)
    try:
        root = project_root_from(Path(__file__))
    except FileNotFoundError:
        raise
    return root / "reports" / "logs" / "asl.log"


def _ensure_log_parent(path: Path) -> None:
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        raise FileNotFoundError(
            f"Không thể ghi log vào {path}: {parent} không phải thư mục."
        )
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Không tìm thấy đường dẫn cha để ghi log: {parent}."
        ) from exc
    except OSError as exc:
        raise FileNotFoundError(f"Không tạo được thư mục log {parent}: {exc}") from exc
    if not parent.is_dir():
        raise FileNotFoundError(f"Thư mục log không tồn tại: {parent}.")


def _coerce_level(level: int | str) -> int:
    if isinstance(level, bool):
        raise ValueError(f"Mức log không hợp lệ: {level!r}.")
    if isinstance(level, int):
        if level not in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ) and (level < 0 or level > 50):
            raise ValueError(f"Mức log không hợp lệ: {level!r}.")
        return level
    if isinstance(level, str):
        numeric = logging.getLevelName(level.upper())
        if isinstance(numeric, int):
            return numeric
    raise ValueError(f"Mức log không hợp lệ: {level!r}.")
