"""Main application entrypoint for EdgeCascade."""

from __future__ import annotations

import argparse
import logging
import sys
import uvicorn

from edgecascade.api.routes import create_app
from edgecascade.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("edgecascade")

app = create_app()


def main() -> None:
    """CLI launcher for EdgeCascade."""
    parser = argparse.ArgumentParser(description="EdgeCascade: Hybrid Inference & Semantic Fallback Router")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--host", type=str, default=None, help="Bind host")
    parser.add_argument("--port", "-p", type=int, default=None, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload for dev")

    args = parser.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port

    logger.info("Starting EdgeCascade on %s:%d", host, port)
    uvicorn.run(
        "edgecascade.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=cfg.server.log_level.lower(),
    )


if __name__ == "__main__":
    main()
