from __future__ import annotations

import logging
from pathlib import Path


class _LoggerSingleton:
    _initialized = False

    @classmethod
    def setup(cls) -> None:
        if cls._initialized:
            return

        log_dir = Path("data")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "log_file.log"

        root = logging.getLogger("alpr")
        root.setLevel(logging.INFO)
        root.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        root.handlers.clear()
        root.addHandler(file_handler)
        root.addHandler(stream_handler)

        cls._initialized = True


def get_logger(name: str = "alpr") -> logging.Logger:
    _LoggerSingleton.setup()
    return logging.getLogger(name)
