import json
import logging
from typing import Any, Dict
from datetime import datetime


class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_action(
        self,
        agent: str,
        action: str,
        arguments: Dict[str, Any],
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "arguments": arguments,
            "state_before": state_before,
            "state_after": state_after,
        }
        self.logger.info(json.dumps(entry))

    def log_info(self, message: str):
        self.logger.info(message)

    def log_error(self, message: str):
        self.logger.error(message)
