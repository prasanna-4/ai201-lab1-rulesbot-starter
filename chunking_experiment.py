"""
Chunking strategy experiment (Optional Challenge 2).

Compares three chunk-size configurations on the SAME documents and the SAME
queries, so we can see where each fails and why:
  - tiny  (chunk_size=80,   overlap=15)
  - base  (chunk_size=300,  overlap=50)   <- the shipped configuration
  - huge  (chunk_size=1200, overlap=100)

It builds a separate IN-MEMORY ChromaDB collection per configuration, so it
never touches the persistent ./chroma_db used by the app. Observations are
written up in planning.md.

Run:  .venv/Scripts/python.exe chunking_experiment.py
"""

import chromadb
from chromadb.utils import embedding_functions
from config import EMBEDDING_MODEL
from ingest import load_documents

CONFIGS = {
    "tiny": {"chunk_size": 80, "overlap": 15, "min_length": 20},
    "base": {"chunk_size": 300, "overlap": 50, "min_length": 50},
    "huge": {"chunk_size": 1200, "overlap": 100, "min_length": 50},
}

QUERIES = [
    "What happens when you roll a 7?",                       # expect Catan
    "How do you get out of Jail in Monopoly?",              # expect Monopoly
    "What happens when you run out of disease cubes?",      # expect Pandemic
    "How does attacking work in Risk?",                     # expect Risk
]

# One shared embedding function across configs so only chunking differs.
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def chunk_with(text, game_name, chunk_size, overlap, min_length):
    chunks = []
    prefix = game_name.lower().replace(" ", "_")
    counter = 0
    start = 0
    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if len(piece) >= min_length:
            chunks.append({
                "text": piece,
                "game": game_name,
                "chunk_id": f"{prefix}_{counter}",
            })
            counter += 1
        start += chunk_size - overlap
    return chunks


def build_collection(name, cfg, documents):
    client = chromadb.EphemeralClient()
    col = client.create_collection(
        name=name, embedding_function=_ef, metadata={"hnsw:space": "cosine"}
    )
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_with(doc["text"], doc["game"], **cfg))
    col.add(
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"game": c["game"]} for c in all_chunks],
        ids=[c["chunk_id"] for c in all_chunks],
    )
    return col, len(all_chunks)


def main():
    documents = load_documents()
    print()
    for label, cfg in CONFIGS.items():
        col, count = build_collection(label, cfg, documents)
        print("#" * 70)
        print(f"CONFIG '{label}'  chunk_size={cfg['chunk_size']} overlap={cfg['overlap']}"
              f"  -> {count} chunks")
        print("#" * 70)
        for q in QUERIES:
            res = col.query(query_texts=[q], n_results=3,
                            include=["documents", "metadatas", "distances"])
            print(f"\nQ: {q}")
            for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                snippet = text.replace("\n", " ")[:75]
                print(f"   [{meta['game']:14}] dist={dist:.3f}  {snippet}")
        print()


if __name__ == "__main__":
    main()
