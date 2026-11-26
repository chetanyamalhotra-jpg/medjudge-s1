"""FastAPI /v1/systemone adapter compatible with Pydantic AI SystemOneModel and Ollama 0.35+."""

from __future__ import annotations

from medjudge.api.app import create_app
from medjudge.api.schemas import SystemOneMode, SystemOneRequest, SystemOneResponse

__all__ = ["SystemOneMode", "SystemOneRequest", "SystemOneResponse", "create_app"]
