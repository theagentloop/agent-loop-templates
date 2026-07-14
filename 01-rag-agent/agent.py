"""
agent.py — The Agent Loop, Build #01
The loop: take a question → retrieve the most relevant chunks from
your documents → send ONLY those chunks to Claude → get a cited answer.

    python agent.py
"""

import os
import sys

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

DB_DIR = "db"
COLLECTION = "documents"
TOP_K = 5                      # how many chunks the agent gets to read
MODEL = "claude-sonnet-4-6"

# The anti-hallucination contract. Every line has a job.
SYSTEM_PROMPT = """You are a document assistant. Answer the user's question using ONLY the provided context.

Rules:
- Base every claim on the context. Do not use outside knowledge.
- After each claim, cite the source in brackets: [filename, page N].
- If the answer is not in the context, say exactly: "I don't have that in the provided documents." Do not guess.
- Be concise. Quote the document's own wording where precision matters."""


def retrieve(collection, question: str) -> list[dict]:
    """Pull the TOP_K most relevant chunks for this question."""
    results = collection.query(query_texts=[question], n_results=TOP_K)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"], "page": meta["page"]})
    return chunks


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['source']}, page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY — copy .env.example to .env and add your key.")

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        sys.exit("No database found. Run: python ingest.py")

    anthropic = Anthropic()
    print("Ask about your documents (Ctrl+C to quit)\n")

    while True:  # ← the agent loop
        try:
            question = input("you > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not question:
            continue

        # 1. RETRIEVE
        chunks = retrieve(collection, question)
        context = build_context(chunks)

        # 2. REASON — Claude sees the question + only your documents
        response = anthropic.messages.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0.2,   # low = factual, repeatable answers
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Context from my documents:\n\n{context}\n\nQuestion: {question}",
            }],
        )

        # 3. RETURN
        print(f"\nagent > {response.content[0].text}\n")
        sources = sorted({f"{c['source']} p.{c['page']}" for c in chunks})
        print(f"        retrieved from: {', '.join(sources)}\n")


if __name__ == "__main__":
    main()
