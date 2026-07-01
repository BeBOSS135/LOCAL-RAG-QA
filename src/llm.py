# Generation Backend
import json
import re

import ollama

import config

# Returned Instead of Generating When the Abstention Gate Fires (Retrieval Too Weak)
# A Fixed String, Not an LLM Call - Faster and Literally Cannot Hallucinate
REFUSAL = "I don't have information about that in my indexed sources."

# Prepended to an Answer That Failed the Post-Generation Grounding Check - We Flag,
# Not Delete, Since the Local Judge Is Imperfect (a False Negative Shouldn't Bin It)
UNVERIFIED_CAVEAT = "⚠️ This answer may not be fully supported by the retrieved sources — verify against them.\n\n"

_VERIFY_PROMPT = (
    "Check whether the ANSWER is fully supported by the CONTEXT. Every factual claim in "
    "the answer must appear in, or directly follow from, the context.\n"
    'Respond with ONLY JSON: {{"supported": true|false, "reason": "<short>"}}\n\n'
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}"
)

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


# Post-Generation Grounding Check: Does the ANSWER Stay Within the CONTEXT? Returns
# (supported, reason). Deterministic (temperature 0). FAIL-OPEN - Returns True on a
# Parse Failure So an Imperfect Local Judge Never Bins a Good Answer Over Bad JSON
def verify(context_blocks: list[str], answer: str) -> tuple[bool, str]:
    client = ollama.Client(host=config.OLLAMA_HOST)
    prompt = _VERIFY_PROMPT.format(context="\n\n".join(context_blocks), answer=answer)
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    )
    match = re.search(r"\{.*\}", resp["message"]["content"], re.DOTALL)
    try:
        data = json.loads(match.group(0))
        return (bool(data["supported"]), str(data.get("reason", ""))[:200])
    except Exception:
        return (True, "verify parse failed (fail-open)")
