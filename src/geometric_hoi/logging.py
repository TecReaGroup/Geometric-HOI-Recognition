"""Daily UTC+08 application logging."""

import logging
from datetime import datetime, timedelta, timezone

from .setting import ROOT

LOCAL_ZONE = timezone(timedelta(hours=8))


class DailyFileSink(logging.Handler):
    """Select the log file using each record's local calendar date."""

    def emit(self, record: logging.LogRecord) -> None:
        """Append a formatted record to today's file."""
        day = datetime.fromtimestamp(record.created, LOCAL_ZONE).strftime("%Y-%m-%d")
        with (ROOT / "log" / f"log_{day}.log").open("a", encoding="utf-8") as stream:
            stream.write(self.format(record) + "\n")


class LocalFormatter(logging.Formatter):
    """Render the required UTC+08 timestamp."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Format a timestamp with a fixed timezone offset."""
        return datetime.fromtimestamp(record.created, LOCAL_ZONE).strftime("%Y-%m-%d %H:%M:%S +08:00")


def configure_logging() -> None:
    """Install console and daily persistent logging."""
    (ROOT / "log").mkdir(exist_ok=True)
    formatter = LocalFormatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for sink in (logging.StreamHandler(), DailyFileSink()):
        sink.setFormatter(formatter)
        root_logger.addHandler(sink)
