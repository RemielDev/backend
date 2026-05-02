import threading
import time
import logging
from collections import defaultdict
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Gemini 2.0 Flash pricing (per 1M tokens)
GEMINI_FLASH_INPUT_PRICE = 0.10   # $0.10 per 1M input tokens
GEMINI_FLASH_OUTPUT_PRICE = 0.40  # $0.40 per 1M output tokens


class TokenTracker:
    """Tracks Gemini API token usage across all agents."""

    def __init__(self):
        self._lock = threading.Lock()
        self._agents: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        )
        self._debug_log: list = []
        self._start_time = time.time()

    def track(self, agent_name: str, input_tokens: int, output_tokens: int):
        with self._lock:
            self._agents[agent_name]["input_tokens"] += input_tokens
            self._agents[agent_name]["output_tokens"] += output_tokens
            self._agents[agent_name]["requests"] += 1
            self._debug_log.append(f"{agent_name}: in={input_tokens} out={output_tokens}")
        logger.info(
            f"Token usage [{agent_name}]: input={input_tokens}, output={output_tokens}"
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_input = sum(a["input_tokens"] for a in self._agents.values())
            total_output = sum(a["output_tokens"] for a in self._agents.values())
            total_requests = sum(a["requests"] for a in self._agents.values())

            input_cost = (total_input / 1_000_000) * GEMINI_FLASH_INPUT_PRICE
            output_cost = (total_output / 1_000_000) * GEMINI_FLASH_OUTPUT_PRICE

            return {
                "agents": dict(self._agents),
                "totals": {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "total_tokens": total_input + total_output,
                    "requests": total_requests,
                },
                "estimated_cost_usd": {
                    "input": round(input_cost, 6),
                    "output": round(output_cost, 6),
                    "total": round(input_cost + output_cost, 6),
                },
                "tracking_since": self._start_time,
                "uptime_seconds": round(time.time() - self._start_time, 1),
            }

    def snapshot(self) -> dict:
        """Capture current totals for later diffing."""
        with self._lock:
            return {
                "agents": {k: dict(v) for k, v in self._agents.items()},
                "debug_pos": len(self._debug_log),
            }

    def diff(self, before: dict) -> Dict[str, Any]:
        """Return tokens used since the snapshot."""
        with self._lock:
            prev_agents = before.get("agents", {})
            prev_debug_pos = before.get("debug_pos", 0)
            agents = {}
            for name, current in self._agents.items():
                prev = prev_agents.get(name, {"input_tokens": 0, "output_tokens": 0, "requests": 0})
                inp = current["input_tokens"] - prev["input_tokens"]
                out = current["output_tokens"] - prev["output_tokens"]
                if inp > 0 or out > 0:
                    agents[name] = {"input_tokens": inp, "output_tokens": out}

            total_input = sum(a["input_tokens"] for a in agents.values())
            total_output = sum(a["output_tokens"] for a in agents.values())
            input_cost = (total_input / 1_000_000) * GEMINI_FLASH_INPUT_PRICE
            output_cost = (total_output / 1_000_000) * GEMINI_FLASH_OUTPUT_PRICE

            return {
                "agents": agents,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "estimated_cost_usd": round(input_cost + output_cost, 8),
                "debug": self._debug_log[prev_debug_pos:],
            }

    def reset(self):
        with self._lock:
            self._agents.clear()
            self._start_time = time.time()


# Singleton
token_tracker = TokenTracker()
