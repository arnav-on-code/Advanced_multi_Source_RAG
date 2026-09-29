import logging
import sys

from app.core.config import settings


LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


def configure_logging() -> None:
    """
    Configure application-wide logging.

    Logs are written to stdout so Docker, AWS EC2,
    and CI/CD platforms can collect them naturally.
    """

    logging.basicConfig(
        level=settings.log_level,
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named application logger."""

    return logging.getLogger(name)