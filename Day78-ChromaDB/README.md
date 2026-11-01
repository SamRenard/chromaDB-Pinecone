# Day 78 - Pinecone Free Tier vs ChromaDB

Runs the same data through Pinecone and ChromaDB using **identical embeddings**
(`all-MiniLM-L6-v2`, 384 dimensions), so differences come from the database and
not the model. It also demonstrates namespaces, metadata filters and index management.

Setup:

1. Create a free Pinecone account and an API key in the console.
2. `pip install -r requirements.txt` (the package is `pinecone`; if the old
   `pinecone-client` is installed, uninstall it first).
3. Set the key: copy `.env.example` to `.env`, or set `PINECONE_API_KEY` in your shell.

```bash
python pinecone_vs_chroma.py
python pinecone_vs_chroma.py --region us-east-1 --k 3 --keep-index
```

Free plans may only allow certain clouds/regions and a limited number of
indexes (often AWS us-east-1); check the Pinecone docs and use `--cloud` /
`--region` if needed. The script deletes its index at the end unless you pass `--keep-index`.

## Concepts

- **Serverless**: you create an index and pay for usage; storage and compute are
  managed and scaled for you (no pods to size).
- **Index**: its `dimension` and `metric` are fixed when it is created. To change
  them, create a new index.
- **Namespaces**: partitions inside one index. Every upsert and query targets one
  namespace, so data in other namespaces is never returned (common for multi-tenant apps).
- **Eventual consistency**: vectors you just upserted may take a moment to appear
  in queries, so the script waits for the counts to match.
- **Metadata filters**: `{"topic": {"$eq": "errors"}}`, also `$in`, `$gte`, `$and`, ...

| | ChromaDB | Pinecone |
|---|---|---|
| Runs | in your process / local disk | managed cloud service |
| Setup | `pip install` | account + API key |
| Scaling | your machine | handled by the service |
| Latency | no network hop | includes a network round trip |
| Cost | free (open source) | free tier, then usage-based |

Key terms: serverless, index, namespace, upsert, dimension, metric, metadata
filter, eventual consistency.
