# electroninja/llm/vision_analyser.py
"""OpenAI‑vision helper using the modern openai‑python ≥ 1.75 client API."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

from electroninja.config.settings import Config
from electroninja.llm.prompts.circuit_prompts import VISION_IMAGE_ANALYSIS_PROMPT

logger = logging.getLogger("electroninja")


class VisionAnalyzer:
    """Analyze circuit PNGs with GPT‑4o (vision)."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.model = self.config.OPENAI_VISION_MODEL  # gpt‑4o
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        logger.info("Vision Analyzer initialised with model: %s", self.model)

    # ------------------------------------------------------------------
    # public: yes/no verification
    # ------------------------------------------------------------------
    def analyze_circuit_image(self, image_path: str, prompt: str) -> str:
        """Return 'Y' or a detailed feedback string."""
        ok, data_uri = self._prepare_image(image_path)
        if not ok:
            return data_uri  # <- data_uri holds an error message

        messages = self._build_messages(prompt, data_uri)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:                        # pragma: no cover
            err = f"Vision analysis error: {exc}"
            logger.error(err, exc_info=True)
            return f"Error: {err}"

    # ------------------------------------------------------------------
    # public: produce description beginning with DESC=
    # ------------------------------------------------------------------
    def produce_description_of_image(self, image_path: str, prompt: str) -> str:
        ok, data_uri = self._prepare_image(image_path)
        if not ok:
            return data_uri

        system_prompt = {
            "role": "system",
            "content": (
                "You are a meticulous circuit‑analysis expert. "
                "Return a textual description that starts with 'DESC='."
            ),
        }
        messages = [system_prompt] + self._build_messages(prompt, data_uri)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            out = resp.choices[0].message.content.strip()
            return out.split("DESC=", 1)[-1].strip() if "DESC=" in out else out
        except Exception as exc:                        # pragma: no cover
            err = f"Vision description error: {exc}"
            logger.error(err, exc_info=True)
            return f"Error: {err}"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _prepare_image(self, path: str) -> tuple[bool, str]:
        """Return (True, data‑URI) or (False, error‑msg)."""
        if not os.path.exists(path):
            err = f"Image file not found: {path}"
            logger.error(err)
            return False, f"Error: {err}"

        size = os.path.getsize(path)
        logger.info("Image file size: %d bytes", size)

        b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"
        return True, data_uri

    @staticmethod
    def _build_messages(prompt: str, data_uri: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                ],
            }
        ]
