# electroninja/llm/prompts/chat_prompts.py


# Circuit request for 4o-mini
CIRCUIT_CHAT_PROMPT = (
    "If the client's message is directly related to circuit design, reply with a concise, confident greeting, "
    "and inform the client that the circuit is being generated. DO NOT include any .asc code in your response. "
    "User prompt: {prompt}\n"
    "Provide a brief, assertive message that assures the client that the circuit is in process."
)

# Non-circuit request for 4o-mini
NON_CIRCUIT_CHAT_PROMPT = (
    "The following request is NOT related to electrical engineering or circuits.\n"
    "User prompt: {prompt}\n"
    "Politely inform the user that you are an electrical engineering assistant and can only help with "
    "circuit design requests. Be courteous but clear about your specific focus area."
)

# Vision feedback response prompt for gpt-4o-mini
VISION_FEEDBACK_PROMPT = (
    "Below is feedback from a vision model about a circuit implementation you are building:\n\n"
    "{vision_feedback}\n\n"
    "Generate a brief, user-friendly response that:\n"
    "1. If the feedback is exactly 'Y', inform the user that their circuit is complete and they can ask for modifications if needed.\n"
    "2. If the feedback contains issues or errors, briefly summarize the main problems identified and assure the user you're working to fix them.\n"
    "In the 2nd case your answer should be of the tone: The current circuit I made has [issues from feedback] but I am working to fix them."
    "Keep your response conversational, concise (2-3 sentences), and non-technical. Do not include any circuit code in your response."
)