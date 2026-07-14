"""
ingest.py — The Agent Loop, Build #01
Reads every document in ./docs, splits it into overlapping chunks,
and stores them in a local ChromaDB vector database (./db).

Run it once, or re-run any time your documents change:
    python ingest.py
"""

import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader

# ── Tuning knobs ─────────────────────────────────────────────
DOCS_DIR = Path("docs")   # put your PDFs / .txt / .md files here
DB_DIR = "db"             # ChromaDB lives in this folder on YOUR disk
CHUNK_WORDS = 600         # ~800 tokens per chunk
OVERLAP_WORDS = 120       # chunks share edges so ideas aren't cut mid-thought
COLLECTION = "documents"


def read_pdf(path: Path) -> list[tuple[str, dict]]:
    """Return a list of (text, metadata) — one entry per page."""
    pages = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, {"source": path.name, "page": i}))
    return pages


def read_text(path: Path) -> list[tuple[str, dict]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, {"source": path.name, "page": 1})] if text.strip() else []


def chunk(text: str, meta: dict) -> list[tuple[str, dict]]:
    """Split text into overlapping word-window chunks, carrying metadata."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [(text, meta)]
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + CHUNK_WORDS])
        if piece.strip():
            chunks.append((piece, dict(meta)))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def main() -> None:
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        sys.exit(f"Put some documents in ./{DOCS_DIR} first (PDF, .txt, or .md).")

    # Fresh database every ingest — simple and predictable.
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    all_chunks: list[tuple[str, dict]] = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            pages = read_pdf(path)
        elif path.suffix.lower() in {".txt", ".md"}:
            pages = read_text(path) if path.name != ".gitkeep" else []
        else:
            continue
        for text, meta in pages:
            all_chunks.extend(chunk(text, meta))
        print(f"  read {path.name}")

    if not all_chunks:
        sys.exit("No readable text found. Scanned PDFs need OCR first.")

    # Chroma's built-in embedding model runs locally — free, no API key.
    collection.add(
        documents=[c[0] for c in all_chunks],
        metadatas=[c[1] for c in all_chunks],
        ids=[f"chunk-{i}" for i in range(len(all_chunks))],
    )
    print(f"\nDone. {len(all_chunks)} chunks stored in ./{DB_DIR}")
    print("Now run: python agent.py")


if __name__ == "__main__":
    main()
