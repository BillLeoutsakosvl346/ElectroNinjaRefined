# electroninja/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralised configuration for ElectroNinja."""

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUTPUT_DIR   = os.path.join(BASE_DIR, "data", "output")
    EXAMPLES_DIR = os.path.join(BASE_DIR, "data", "examples_asc")

    # ------------------------------------------------------------------
    # LT‑Spice
    # ------------------------------------------------------------------
    # 1) honour env‑var if set
    # 2) otherwise build the per‑user path:  C:\Users\<YOU>\AppData\Local\Programs\ADI\LTspice\LTspice.exe
    # 3) you can still override with LTSPICE_PATH in .env
    LTSPICE_PATH = os.getenv(
        "LTSPICE_PATH",
        os.path.join(os.path.expanduser("~"), r"AppData\Local\Programs\ADI\LTspice\LTspice.exe")
    )

    # ------------------------------------------------------------------
    # OpenAI models (unchanged)
    # ------------------------------------------------------------------
    OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
    ASC_MODEL          = os.getenv("ASC_MODEL", "o4-mini")
    CHAT_MODEL         = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    EVALUATION_MODEL   = os.getenv("EVALUATION_MODEL", "gpt-4o-mini")
    DESCRIPTION_MODEL  = os.getenv("DESCRIPTION_MODEL", "gpt-4o-mini")
    MERGER_MODEL       = os.getenv("MERGER_MODEL", "gpt-4o-mini")
    COMPONENT_MODEL    = os.getenv("COMPONENT_MODEL", "gpt-4o-mini")
    OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

    # ------------------------------------------------------------------
    # Vector‑DB
    # ------------------------------------------------------------------
    VECTOR_DB_DIR      = os.path.join(BASE_DIR, "data", "vector_db")
    VECTOR_DB_INDEX    = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
    VECTOR_DB_METADATA = os.path.join(VECTOR_DB_DIR, "metadata_list.pkl")

    # ------------------------------------------------------------------
    # create dirs on first import
    # ------------------------------------------------------------------
    @classmethod
    def ensure_directories(cls) -> None:
        os.makedirs(cls.OUTPUT_DIR,    exist_ok=True)
        os.makedirs(cls.VECTOR_DB_DIR, exist_ok=True)
