# Generation Backend
import ollama

import config

# Instructed to Stay Grounded So We Can Measure Faithfulness Later
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


# Send Context + Question to the Local LLM, Return the Full Answer
def generate(question: str, context_blocks: list[str]) -> str:
    client = ollama.Client(host=config.OLLAMA_HOST)
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=_messages(question, context_blocks),
        options={"temperature": 0.1},      # Low Temp -> Stick to the Context
        keep_alive=config.OLLAMA_KEEP_ALIVE,  # Stay Resident in VRAM Between Queries
    )
    return resp["message"]["content"].strip()


# Yield the Answer Token-by-Token as It Generates (for Responsive UIs)
def generate_stream(question: str, context_blocks: list[str]):
    client = ollama.Client(host=config.OLLAMA_HOST)
    for part in client.chat(
        model=config.OLLAMA_MODEL,
        messages=_messages(question, context_blocks),
        options={"temperature": 0.1},
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        stream=True,
    ):
        yield part["message"]["content"]
