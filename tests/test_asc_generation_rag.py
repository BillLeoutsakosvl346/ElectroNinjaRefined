"""
tests/test_asc_generation_rag.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*   Loads the description produced for prompt 1.
*   Runs CircuitGenerator.generate_asc_code().
*   Prints the full user‑prompt that is sent to the ASC‑generation model
    (detected by model‑ID == provider.asc_gen_model).
*   Verifies that the returned ASC code begins with “Version 4”.
"""

import os
import sys
import logging
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

# ── project imports ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from electroninja.config.settings import Config
from electroninja.llm.providers.openai import OpenAIProvider
from electroninja.llm.vector_store import VectorStore
from electroninja.backend.circuit_generator import CircuitGenerator

# ── env / logging ──────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

PROMPT_ID = 1  # data/output/prompt1/…

# ----------------------------------------------------------------------
# monkey‑patch helpers
# ----------------------------------------------------------------------
def _make_fake_chat_completion(target_model: str):
    """Factory so we can compare against `provider.asc_gen_model` via closure."""

    def fake_chat_completion(*, model: str, messages, **kwargs):
        # Only print the prompt for the ASC‑generation call
        if model == target_model:
            border = "=" * 60
            print(f"\n{border}\nPROMPT SENT TO ASC MODEL ({model})\n{border}")
            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                print(f"{role}:\n{content}\n{'-'*40}")

        # Minimal ChatCompletion‑like return with dummy ASC code
        dummy_asc = (
            "Version 4\n"
            "SHEET 1 880 680\n"
            "V1 0 1 DC 3V\n"
            "R1 1 2 4\n"
            "R2 1 2 4\n"
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=dummy_asc))]
        )

    return fake_chat_completion


def fake_embedding(*args, **kwargs):
    """Return a zero vector with correct shape."""
    zeros = [0.0] * 1536
    return SimpleNamespace(data=[SimpleNamespace(embedding=zeros)])


# ----------------------------------------------------------------------
# test body
# ----------------------------------------------------------------------
def test_asc_generation_from_description():
    desc_path = ROOT / "data" / "output" / f"prompt{PROMPT_ID}" / "description.txt"
    assert desc_path.exists(), f"Missing description file: {desc_path}"
    description = desc_path.read_text(encoding="utf-8").strip()

    print("\n=== LOADED DESCRIPTION ===")
    print(description)

    # initialise core objects
    config = Config()
    provider = OpenAIProvider(config)
    vector_store = VectorStore(config)
    circuit_gen = CircuitGenerator(provider, vector_store)

    # monkey‑patch endpoints (after objects exist)
    provider.client.chat.completions.create = _make_fake_chat_completion(
        provider.asc_gen_model
    )
    vector_store.client.embeddings.create = fake_embedding

    # run generator
    asc_code = circuit_gen.generate_asc_code(description, PROMPT_ID)

    print("\n=== ASC CODE RETURNED BY GENERATOR ===")
    print(asc_code)

    assert asc_code.startswith("Version 4"), "ASC output malformed"


if __name__ == "__main__":
    test_asc_generation_from_description()
