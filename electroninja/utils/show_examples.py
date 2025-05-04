# electroninja/utils/show_top_examples.py
"""
Demo utility: search the vector store for a hard‑coded query and print the
three most similar example circuits (description + ASC code snippet).

How to run
~~~~~~~~~~
python electroninja/utils/show_top_examples.py
"""

from __future__ import annotations

from textwrap import indent

from electroninja.config.settings import Config
from electroninja.llm.vector_store import VectorStore

# ----------------------------------------------------------------------
# you can change this to any query you like
DEMO_QUERY: str = "A low‑pass RC filter"
TOP_K: int = 3
SNIPPET_LEN: int = 800  # chars of ASC code to show
# ----------------------------------------------------------------------


def main() -> None:
    print(f"\n🔍  Query: “{DEMO_QUERY}”\n")

    # 1) load vector store
    vs = VectorStore(Config())
    if not vs.load():
        print("❌  Could not load the FAISS index. "
              "Run ingest_examples.py first if needed.")
        return

    # 2) retrieve examples
    hits = vs.search(DEMO_QUERY, top_k=TOP_K)
    if not hits:
        print("No similar examples found.\n")
        return

    # 3) pretty‑print results
    for rank, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})
        desc = meta.get("description", "No description available")
        asc  = hit.get("asc_code", "").strip()
        dist = hit.get("score", 0.0)

        print(f"─── Example {rank}  (distance {dist:.4f}) ───────────────")
        print(f"Description:\n{indent(desc, '  ')}\n")
        print("ASC code snippet:")
        snippet = asc[:SNIPPET_LEN] + ("…\n" if len(asc) > SNIPPET_LEN else "\n")
        print(indent(snippet, "  "))
        print("──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
