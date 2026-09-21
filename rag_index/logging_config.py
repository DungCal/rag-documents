import logging
import logging.handlers
import sys
from pathlib import Path

from . import config


def setup_logging() -> logging.Logger:
    """Configure the shared rag_index logger: console always, file optional."""
    logger = logging.getLogger("rag_index")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if config.LOG_TO_FILE and config.LOG_FILE:
        log_path = Path(config.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def setup_query_logging() -> logging.Logger:
    """Configure the query-audit logger: one raw JSON line per query, file only."""
    q_logger = logging.getLogger("rag_index.query")
    q_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    q_logger.propagate = False
    q_logger.handlers.clear()

    if not config.QUERY_LOG_FILE:
        return q_logger

    log_path = Path(config.QUERY_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(message)s"))
    q_logger.addHandler(fh)

    return q_logger


logger = setup_logging()
query_logger = setup_query_logging()