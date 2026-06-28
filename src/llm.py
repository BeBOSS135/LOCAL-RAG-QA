"""Generation backend. Pluggable: today Ollama, swap here for an API later."""
import ollama

import config

# Instructed to stay grounded so we can measure faithfulness later
SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the provided "
    "context. If the context does not contain the answer, say you don't know. "
    "Be concise and cite sources by their [source] tags."
)


def _messages(question: str, context_blocks: list[str]) -> list[dict]:
    context = "\n\n".join(context_blocks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate(question: str, context_blocks: list[str]) -> str:
    """Send context + question to the local LLM and return the full answer text."""
    client = ollama.Client(host=config.OLLAMA_HOST)
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=_messages(question, context_blocks),
        options={"temperature": 0.1},      # low temp -> stick to the context
        keep_alive=config.OLLAMA_KEEP_ALIVE,  # stay resident in VRAM between queries
    )
    return resp["message"]["content"].strip()


def generate_stream(question: str, context_blocks: list[str]):
    """Yield the answer token-by-token as it's generated (for responsive UIs)."""
    client = ollama.Client(host=config.OLLAMA_HOST)
    for part in client.chat(
        model=config.OLLAMA_MODEL,
        messages=_messages(question, context_blocks),
        options={"temperature": 0.1},
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        stream=True,
    ):
        yield part["message"]["content"]
