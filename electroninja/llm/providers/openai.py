# electroninja/llm/providers/openai.py
"""
OpenAI Provider (SDK ≥ 1.75)

* Uses the modern ``OpenAI`` client (no deprecated helpers).
* ASC generation / refinement model defaults to **o4‑mini** (overridable by
  ``ASC_MODEL`` env var).
* Fulfils every abstract method declared in ``LLMProvider``.
* `_build_prompt()` injects at most three RAG examples.
* `generate_asc_code()` prints the final prompt so tests or live runs can
  inspect what goes into the model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI

from electroninja.config.settings import Config
from electroninja.llm.providers.base import LLMProvider
from electroninja.llm.prompts.circuit_prompts import (
    ASC_SYSTEM_PROMPT,
    ASC_REFINEMENT_PROMPT_TEMPLATE,
    CIRCUIT_RELEVANCE_EVALUATION_PROMPT,
    DESCRIPTION_PROMPT,
)
from electroninja.llm.prompts.chat_prompts import (
    CIRCUIT_CHAT_PROMPT,
    VISION_FEEDBACK_PROMPT,
)

logger = logging.getLogger("electroninja")


# ======================================================================
# provider
# ======================================================================
class OpenAIProvider(LLMProvider):
    """Concrete implementation that communicates with the OpenAI API."""

    # ------------------------------------------------------------------
    # ctor
    # ------------------------------------------------------------------
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)

        # model ids (overridable via env vars)
        self.asc_gen_model: str = self.config.ASC_MODEL          # o4‑mini
        self.chat_model: str = self.config.CHAT_MODEL
        self.evaluation_model: str = self.config.EVALUATION_MODEL
        self.description_model: str = self.config.DESCRIPTION_MODEL
        self.merger_model: str = self.config.MERGER_MODEL

        self.logger = logger

    # ------------------------------------------------------------------
    # low‑level helper
    # ------------------------------------------------------------------
    def _chat_complete(
        self, *, model: str, messages: List[Dict[str, str]]
    ) -> str:
        """Call the OpenAI chat endpoint and return the first choice’s content."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # step 1 – circuit relevance evaluation
    # ------------------------------------------------------------------
    def evaluate_circuit_request(self, prompt: str) -> str:
        try:
            return self._chat_complete(
                model=self.evaluation_model,
                messages=[{
                    "role": "user",
                    "content": CIRCUIT_RELEVANCE_EVALUATION_PROMPT.format(prompt=prompt),
                }],
            )
        except Exception as exc:                       # pragma: no cover
            self.logger.error("Evaluation error: %s", exc, exc_info=True)
            return "N"

    # ------------------------------------------------------------------
    # step 2a – description creation
    # ------------------------------------------------------------------
    def create_description(
        self, previous_description: str, new_request: str
    ) -> str:
        prompt = DESCRIPTION_PROMPT.format(
            previous_description=previous_description or "None",
            new_request=new_request,
        )
        try:
            return self._chat_complete(
                model=self.description_model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:                       # pragma: no cover
            self.logger.error("Description error: %s", exc, exc_info=True)
            return new_request

    # ------------------------------------------------------------------
    # step 3 – ASC generation
    # ------------------------------------------------------------------
    def generate_asc_code(
        self,
        description: str,
        examples: Optional[List[Dict[str, str]]] = None,
        prompt_id: Optional[int] = None,
    ) -> str:
        user_prompt = self._build_prompt(description, examples, prompt_id)

        # ↓↓↓ debug – always show the prompt sent to o4‑mini ↓↓↓
        print("\n" + "=" * 60,
              "\nPROMPT SENT TO ASC MODEL (o4‑mini)\n",
              "=" * 60, sep="")
        print(user_prompt[:1500] + ("…" if len(user_prompt) > 1500 else ""))
        print("=" * 60 + "\n")

        try:
            raw = self._chat_complete(
                model=self.asc_gen_model,
                messages=[
                    {"role": "system", "content": ASC_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return "N" if raw.upper() == "N" else self.extract_clean_asc_code(raw)
        except Exception as exc:                       # pragma: no cover
            self.logger.error("ASC generation error: %s", exc, exc_info=True)
            return "Error: failed to generate circuit"

    # ------------------------------------------------------------------
    # step 6b – ASC refinement
    # ------------------------------------------------------------------
    def refine_asc_code(
        self, prompt_id: int, iteration: int, vision_feedback: str
    ) -> str:
        prompt = ASC_REFINEMENT_PROMPT_TEMPLATE.format(
            prompt_id=prompt_id,
            iteration=iteration,
            vision_feedback=vision_feedback,
        )
        try:
            return self._chat_complete(
                model=self.asc_gen_model,
                messages=[
                    {"role": "system", "content": ASC_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
        except Exception as exc:                       # pragma: no cover
            self.logger.error("Refinement error: %s", exc, exc_info=True)
            return "Error refining ASC code"

    # ------------------------------------------------------------------
    # 2b & 6a – chat helpers required by base class
    # ------------------------------------------------------------------
    def generate_chat_response(self, prompt: str) -> str:
        """User‑facing chat bubble after initial request or vision feedback."""
        wrapped = CIRCUIT_CHAT_PROMPT.format(prompt=prompt)
        try:
            return self._chat_complete(
                model=self.chat_model,
                messages=[{"role": "user", "content": wrapped}],
            )
        except Exception as exc:                       # pragma: no cover
            self.logger.error("Chat response error: %s", exc, exc_info=True)
            return "Error generating chat response"

    def generate_vision_feedback_response(self, feedback: str) -> str:
        """Turn raw vision‑model feedback into a polite chat bubble."""
        wrapped = VISION_FEEDBACK_PROMPT.format(vision_feedback=feedback)
        try:
            return self._chat_complete(
                model=self.chat_model,
                messages=[{"role": "user", "content": wrapped}],
            )
        except Exception as exc:                       # pragma: no cover
            self.logger.error("Vision feedback chat error: %s", exc, exc_info=True)
            return "Error generating vision feedback response"

    # ------------------------------------------------------------------
    # prompt builder – merges instructions, examples, and description
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        description: str,
        examples: Optional[List[Dict[str, str]]],
        prompt_id: Optional[int],
    ) -> str:
        load = self._load_instruction
        parts: List[str] = []

        # 1) always‑included instructions
        parts.append("=== GENERAL INSTRUCTIONS ===\n"  + load("general_instruct.txt")  + "\n")
        parts.append("=== BATTERY INSTRUCTIONS ===\n"  + load("battery_instruct.txt")  + "\n")

        # 2) component‑specific instructions
        if prompt_id is not None:
            comp_file = Path("data/output") / f"prompt{prompt_id}" / "components.txt"
            if comp_file.is_file():
                letters = comp_file.read_text(encoding="utf-8").upper()
                if "R" in letters:
                    parts.append("=== RESISTOR INSTRUCTIONS ===\n"  + load("resistor_instruct.txt")  + "\n")
                if "C" in letters:
                    parts.append("=== CAPACITOR INSTRUCTIONS ===\n" + load("capacitor_instruct.txt") + "\n")
                if "L" in letters:
                    parts.append("=== INDUCTOR INSTRUCTIONS ===\n"  + load("inductor_instruct.txt")  + "\n")
                if "D" in letters:
                    parts.append("=== DIODE INSTRUCTIONS ===\n"     + load("diode_instruct.txt")     + "\n")

        # 3) RAG examples (first three only)
        if examples:
            parts.append("=== SIMILAR EXAMPLES ===")
            for i, ex in enumerate(examples[:3], 1):
                meta = ex.get("metadata", {})
                desc = meta.get("description", "No description available")
                asc  = ex.get("asc_code", "")
                parts.append(f"\nExample {i}:\nDescription: {desc}\nASC Code:\n{asc}\n")

        # 4) user description
        parts.append("=== CIRCUIT DESCRIPTION ===\n" + description + "\n")

        # 5) final directive
        parts.append("=== TASK ===\nReturn **only** the .asc file content.")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # tiny utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _load_instruction(filename: str) -> str:
        path = Path("electroninja/llm/prompts/instructions") / filename
        return path.read_text(encoding="utf-8")

    @staticmethod
    def extract_clean_asc_code(raw: str) -> str:
        """Unwrap ``` fences if present."""
        raw = raw.strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        return raw
