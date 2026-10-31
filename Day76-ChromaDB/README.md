# Day 76 - ChromaDB (and Pinecone)

A runnable ChromaDB walkthrough: create a collection, add documents with
metadata, query by meaning, filter by metadata, compare the index's answer with
exact search, then get / upsert / delete.

```bash
pip install -r requirements.txt
python chroma_demo.py          # fresh collection every run
python chroma_demo.py --keep   # reuse the saved collection (persistence)
```

The first run downloads Chroma's default small embedding model.
Chroma defaults to L2 distance, so the script asks for cosine
(`metadata={"hnsw:space": "cosine"}`); then distance = 1 - cosine similarity.
If your Chroma version rejects that option, check the docs for the current way
to set the distance function.

## How vector databases search

Exact search compares the query with every stored vector (cost grows with the
number of vectors). **ANN (approximate nearest neighbor)** search trades a small
loss of accuracy for a big speedup.

**HNSW** builds a layered graph: the top layers are sparse and allow long jumps,
the bottom layer contains every vector. A search enters at the top, moves
greedily toward the query, drops one layer at a time, and explores close
neighbors in the bottom layer. Main knobs: `M` (links per node), `ef_construction`
(build quality) and `ef_search` (search breadth; higher = better recall, slower).

## Chroma vs Pinecone

| | ChromaDB | Pinecone |
|---|---|---|
| Runs | locally / embedded in your app | managed cloud service (API key) |
| Good for | prototypes, local projects | production scale without running servers |

Key terms: vector database, collection, ANN, HNSW, metadata filter, upsert,
distance metric, persistence.
