"""
End‑to‑end RAG test that actually calls o4‑mini.

* Loads prompt 1 description.
* Loads FAISS index so real examples are returned.
* Wraps provider._chat_complete to print the prompt, then calls the real API.
* Asserts the returned ASC file starts with “Version 4”.

CAUTION: consumes OpenAI credits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import wraps
from typing import List, Dict

# ── repo imports (ensure root on path) ─────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from electroninja.config.settings import Config                           # noqa: E402
from electroninja.llm.vector_store import VectorStore                      # noqa: E402
from electroninja.llm.providers.openai import OpenAIProvider               # noqa: E402
from electroninja.backend.circuit_generator import CircuitGenerator        # noqa: E402

PROMPT_ID = 1
DESC_PATH = Path("data/output") / f"prompt{PROMPT_ID}" / "description.txt"

assert DESC_PATH.exists(), f"Missing description file: {DESC_PATH}"
description = DESC_PATH.read_text(encoding="utf-8").strip()
print("\nLoaded description:\n" + description + "\n")

# ── set up objects ────────────────────────────────────────────────────
config = Config()
vector_store = VectorStore(config)
if not vector_store.load():
    raise RuntimeError("Vector store failed to load. Run ingest_examples.py.")

provider = OpenAIProvider(config)
circuit_gen = CircuitGenerator(provider, vector_store)

# ── wrap _chat_complete so we can see the prompt, but still hit OpenAI ─
orig_chat_complete = provider._chat_complete


@wraps(orig_chat_complete)
def verbose_chat_complete(*args, **kwargs):
    model: str = kwargs.get("model") or args[1]  # type: ignore[index]
    messages: List[Dict[str, str]] = kwargs.get("messages") or args[2]  # type: ignore[index]

    print("\n" + "=" * 80)
    print(f"PROMPT BUNDLE SENT TO MODEL  ({model})")
    print("=" * 80 + "\n")
    for m in messages:
        role = m["role"].upper()
        print(f"--- {role} ---\n{m['content']}\n")
    print("=" * 80 + "\n")

    # call the real API
    return orig_chat_complete(*args, **kwargs)


provider._chat_complete = verbose_chat_complete  # type: ignore[method-assign]

# ── run generation ────────────────────────────────────────────────────
asc_code = circuit_gen.generate_asc_code(description, PROMPT_ID)

print("\nReturned ASC code:\n", asc_code[:1000])
assert asc_code.startswith("Version 4"), "ASC code should start with 'Version 4'"
print("✅  Live call succeeded")
