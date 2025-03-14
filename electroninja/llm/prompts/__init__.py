# electroninja/llm/prompts/__init__.py

from electroninja.llm.prompts.circuit_prompts import (
    GENERAL_INSTRUCTION,
    SAFETY_FOR_AGENT,
    ASC_GENERATION_PROMPT,
    REFINEMENT_PROMPT_TEMPLATE,
    VISION_ANALYSIS_PROMPT_TEMPLATE,
    VISION_IMAGE_ANALYSIS_PROMPT,
    CIRCUIT_RELEVANCE_EVALUATION_PROMPT,
    RAG_ASC_GENERATION_PROMPT
)

from electroninja.llm.prompts.chat_prompts import (
    CIRCUIT_CHAT_PROMPT,
    NON_CIRCUIT_CHAT_PROMPT,
    VISION_FEEDBACK_PROMPT
)

__all__ = [
    "GENERAL_INSTRUCTION",
    "SAFETY_FOR_AGENT",
    "ASC_GENERATION_PROMPT",
    "REFINEMENT_PROMPT_TEMPLATE",
    "VISION_ANALYSIS_PROMPT_TEMPLATE",
    "VISION_IMAGE_ANALYSIS_PROMPT",
    "CIRCUIT_RELEVANCE_EVALUATION_PROMPT",
    "RAG_ASC_GENERATION_PROMPT",
    "GREETING_PROMPT",
    "CIRCUIT_REQUEST_TEMPLATE",
    "DESIGN_IN_PROGRESS_TEMPLATE",
    "ITERATION_STATUS_TEMPLATE",
    "SUCCESS_TEMPLATE",
    "FAILURE_TEMPLATE",
    "GENERAL_HELP_RESPONSE",
    "LTSPICE_EXPLANATION",
    "CIRCUIT_CHAT_PROMPT",
    "NON_CIRCUIT_CHAT_PROMPT",
    "VISION_FEEDBACK_PROMPT"
]