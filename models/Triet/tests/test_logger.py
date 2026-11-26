"""Kiểm tra logger ghi file và báo lỗi đường dẫn."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.utils.logger import get_logger, setup_logger


def test_setup_logger_writes_console_and_file(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "asl.log"
    logger = setup_logger("asl.test.write", log_file=log_file, level="INFO")
    logger.info("dong nhat ky")
    for handler in logger.handlers:
        handler.flush()

    assert log_file.is_file()
    text = log_file.read_text(encoding="utf-8")
    assert "dong nhat ky" in text
    assert any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
    kinds = [type(handler) for handler in logger.handlers]
    assert kinds.count(logging.FileHandler) + sum(
        issubclass(kind, logging.FileHandler) for kind in kinds
    ) >= 1

    again = setup_logger("asl.test.write", log_file=log_file, level="DEBUG")
    assert len(again.handlers) == 2
    assert get_logger("asl.test.write") is again


def test_log_parent_not_a_directory_raises_file_not_found(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file", encoding="utf-8")
    log_file = blocker / "asl.log"

    with pytest.raises(FileNotFoundError):
        setup_logger("asl.test.blocked", log_file=log_file)
