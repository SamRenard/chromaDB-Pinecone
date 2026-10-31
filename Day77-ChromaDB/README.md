# Day 77 - ChromaDB: Metadata Filtering and Persistence with PDFs

Stores PDF text in ChromaDB as chunks with metadata (`source`, `page`, `chunk`)
and searches them with filters.

```bash
pip install -r requirements.txt
python pdf_to_chroma.py index --demo                       # no PDF needed
python pdf_to_chroma.py index --pdf manual.pdf guide.pdf   # your own PDFs
python pdf_to_chroma.py stats
python pdf_to_chroma.py search "how do I reset it" --source router_manual.pdf
python pdf_to_chroma.py search "payment" --page-min 2 --page-max 3
python pdf_to_chroma.py search "warranty" --contains two
```

The database is saved in `./chroma_db`, so you can search in a later run without
indexing again. Indexing uses stable ids (`file-p<page>-c<chunk>`) and `upsert`,
so running it twice does not create duplicates.

## What to know

- **Metadata filters** (`where`): `{"source": "a.pdf"}`, `{"page": {"$gte": 2}}`,
  `{"source": {"$in": [...]}}`. Operators: `$eq $ne $gt $gte $lt $lte $in $nin`.
  Several conditions must be wrapped in `{"$and": [...]}`. Numeric filters only
  work if the value was stored as a number (page is an int).
- **Text filter** (`where_document`): `{"$contains": "text"}` filters by the chunk text.
- **Persistence**: `PersistentClient(path=...)` stores data on disk;
  `chromadb.Client()` is in-memory only.
- **Embedding functions**: Chroma embeds documents and queries with the
  collection's embedding function. Always reopen a collection with the same one it
  was built with, or query vectors land in a different space. The script saves the
  embedder name in the collection metadata and reuses it automatically.
  `--embedder all-MiniLM-L6-v2` uses a sentence-transformers model instead of the default.
- Scanned PDFs have no extractable text and would need OCR.

Key terms: metadata filtering, where, where_document, persistence, upsert,
embedding function, collection.
