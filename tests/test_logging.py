from __future__ import annotations

import logging

from app.core.logging import (
    configure_logging,
    get_logger,
)


def test_get_logger():
    logger = get_logger("test_logger")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_get_logger_default_name():
    logger = get_logger()

    assert isinstance(logger, logging.Logger)
    assert logger.name


def test_configure_logging():
    configure_logging()

    logger = get_logger("p1_test")

    assert isinstance(logger, logging.Logger)

    logger.info("Logging configuration test")


def test_logger_can_log(caplog):
    logger = get_logger("p1_test_logging")

    with caplog.at_level(logging.INFO):
        logger.info("Test log message")

    assert "Test log message" in caplog.text