# Sample Document

This is a placeholder document so the RAG pipeline runs out of the box. Replace it
with your own `.txt`, `.md`, or `.pdf` files in this `data/` folder, then run
`python index.py`.

## Retrieval-Augmented Generation

RAG answers questions about your own documents by retrieving the most relevant text
chunks and giving them to a language model as context, so the answer is grounded in
your sources rather than the model's general training data.

## How this project works

Documents are split into overlapping, token-bounded chunks and embedded into vectors.
At query time the question is embedded too, the closest chunks are retrieved (vector
search combined with BM25 keyword search), a cross-encoder re-ranks them, and a local
LLM (Mistral via Ollama) writes a grounded answer with source citations.

## Why local

Everything runs offline on a consumer GPU: the embedding model, the re-ranker, and
the LLM all run on your own machine, so no documents or questions leave it.
