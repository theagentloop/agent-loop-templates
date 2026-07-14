# Build #01 — RAG Agent Over Your Own Files

An AI agent that answers questions from YOUR documents — with citations —
and says "I don't know" instead of making things up.

From [The Agent Loop](https://www.youtube.com/@theagentloop). Full build video: [link]

## Quickstart (5 steps)

1. **Install** (Python 3.10+)
   ```
   pip install -r requirements.txt
   ```
2. **Add your API key** — get one at console.anthropic.com
   ```
   cp .env.example .env    # then paste your key into .env
   ```
3. **Add documents** — drop PDFs, .txt, or .md files into `docs/`
4. **Build the memory**
   ```
   python ingest.py
   ```
5. **Ask questions**
   ```
   python agent.py
   ```

## How it works

`ingest.py` splits your documents into ~800-token overlapping chunks and stores
them (with filename + page metadata) in a local ChromaDB folder. Embeddings run
locally — free, no API calls. `agent.py` runs the loop: your question → retrieve
the 5 most relevant chunks → Claude answers from ONLY those chunks → cited answer.

Your files never leave your machine except the specific chunks sent to Claude
for the question you asked.

## Limits (honest ones)

- Scanned/image PDFs need OCR first (`ocrmypdf` works)
- No chat memory yet — every question stands alone (that's a future build)
- Very large document sets want smarter chunking

## New here?

One complete agent build every Sunday, template always free:
**[youtube.com/@theagentloop](https://www.youtube.com/@theagentloop)**
