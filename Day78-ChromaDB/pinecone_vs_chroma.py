"""Day 78 - Pinecone (free tier) vs ChromaDB on the same data and the SAME vectors.

Both databases receive identical embeddings, so any difference comes from the database,
not the model. Also demonstrates namespaces, metadata filters and index management.

    export PINECONE_API_KEY=...        (Windows PowerShell: $env:PINECONE_API_KEY="...")
    python pinecone_vs_chroma.py
"""
import argparse
import os
import statistics
import sys
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL = "all-MiniLM-L6-v2"  # 384-dimensional embeddings

DOCS = [  # (text, topic)
    ("To reset your password, click 'Forgot password' on the sign-in page and follow the email link.", "account"),
    ("Refunds are issued within 14 days of purchase if the item arrives damaged or defective.", "orders"),
    ("Standard shipping takes 3 to 5 business days; express shipping arrives the next day.", "orders"),
    ("Error ERR_4021 means the payment token has expired; request a new checkout session.", "errors"),
    ("Our store is open Monday to Saturday from 9 AM to 6 PM.", "store"),
    ("You can change your subscription plan at any time from the billing settings.", "billing"),
    ("Two-factor authentication adds a second verification step when you sign in.", "account"),
    ("To delete your account permanently, contact support and confirm your identity.", "account"),
    ("The mobile app supports offline mode, syncing your changes when you reconnect.", "app"),
    ("Gift cards never expire and can be used for any product in the store.", "store"),
    ("Error ERR_5003 indicates that the server could not reach the database.", "errors"),
    ("Invoices are emailed on the first day of every month and can be downloaded as PDF.", "billing"),
]
OTHER_DOCS = [  # go into a second namespace to show isolation
    ("Preheat the oven to 180 degrees and bake the bread for 30 minutes.", "cooking"),
    ("Mix flour, water and yeast, then let the dough rise for one hour.", "cooking"),
]
QUERIES = [
    "I can't log into my account",
    "money back for a broken product",
    "how long until my package arrives",
    "the server can't reach the database",
]


# ------------------------------------------------------------ pure helpers --

def to_pinecone_vectors(ids, vectors, docs):
    """Pinecone record format: id + values + metadata (we keep the text and topic as metadata)."""
    return [{"id": i, "values": list(map(float, v)), "metadata": {"text": text, "topic": topic}}
            for i, v, (text, topic) in zip(ids, vectors, docs)]


def overlap_at_k(ids_a, ids_b):
    """Share of ids that both result lists have in common (order ignored)."""
    if not ids_a and not ids_b:
        return 1.0
    return len(set(ids_a) & set(ids_b)) / max(len(set(ids_a)), len(set(ids_b)))


def median_ms(fn, repeats=5):
    fn()  # warm-up call, not timed
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def is_ready(pc, name):
    status = pc.describe_index(name).status
    try:
        return bool(status["ready"])
    except (TypeError, KeyError):
        return bool(getattr(status, "ready", False))


def stats_dict(index):
    stats = index.describe_index_stats()
    return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)


def wait_for_count(index, namespace, expected, timeout=60):
    """Serverless indexes are eventually consistent: wait until the upserted vectors are visible."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        count = stats_dict(index).get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
        if count >= expected:
            return True
        time.sleep(1)
    return False


# ----------------------------------------------------------------- engines --

def setup_pinecone(pc, name, dimension, cloud, region):
    from pinecone import ServerlessSpec

    if name in pc.list_indexes().names():
        existing = pc.describe_index(name).dimension
        if existing != dimension:
            sys.exit(f"Index '{name}' already exists with dimension {existing}, need {dimension}. "
                     "Dimension and metric cannot be changed: pick another --index name or delete it.")
        print(f"Reusing existing index '{name}'.")
    else:
        print(f"Creating serverless index '{name}' ({cloud}/{region}, dimension {dimension}, cosine)...")
        pc.create_index(name=name, dimension=dimension, metric="cosine",
                        spec=ServerlessSpec(cloud=cloud, region=region))
    while not is_ready(pc, name):
        time.sleep(1)
    return pc.Index(name)


def query_pinecone(index, vector, k, namespace, flt=None):
    response = index.query(vector=list(map(float, vector)), top_k=k, namespace=namespace, filter=flt)
    return [m.id for m in response.matches], [m.score for m in response.matches]


def query_chroma(collection, vector, k, where=None):
    result = collection.query(query_embeddings=[list(map(float, vector))], n_results=k, where=where)
    return result["ids"][0], [1 - d for d in result["distances"][0]]  # cosine distance -> similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="day78-compare")
    parser.add_argument("--cloud", default="aws")
    parser.add_argument("--region", default="us-east-1",
                        help="Free plans may only allow certain cloud/regions; see Pinecone docs")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--keep-index", action="store_true", help="Do not delete the Pinecone index at the end")
    args = parser.parse_args()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        sys.exit("Set the PINECONE_API_KEY environment variable (create a key in the Pinecone console).")

    import chromadb
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer

    # ---- identical embeddings for both databases ---- #
    texts = [t for t, _ in DOCS] + [t for t, _ in OTHER_DOCS] + QUERIES
    vectors = SentenceTransformer(MODEL).encode(texts, normalize_embeddings=True)
    doc_vecs = vectors[:len(DOCS)]
    other_vecs = vectors[len(DOCS):len(DOCS) + len(OTHER_DOCS)]
    query_vecs = vectors[len(DOCS) + len(OTHER_DOCS):]
    ids = [f"doc-{i}" for i in range(len(DOCS))]
    other_ids = [f"other-{i}" for i in range(len(OTHER_DOCS))]
    dimension = vectors.shape[1]

    # ---- ChromaDB (local, in memory) ---- #
    chroma = chromadb.Client().create_collection("compare", metadata={"hnsw:space": "cosine"})
    chroma.add(ids=ids, embeddings=[list(map(float, v)) for v in doc_vecs],
               documents=[t for t, _ in DOCS], metadatas=[{"topic": p} for _, p in DOCS])

    # ---- Pinecone (managed cloud) ---- #
    pc = Pinecone(api_key=api_key)
    index = setup_pinecone(pc, args.index, dimension, args.cloud, args.region)
    info = pc.describe_index(args.index)
    print(f"Index: dimension={info.dimension}, metric={info.metric}, host={info.host}")

    namespace = "support"
    index.upsert(vectors=to_pinecone_vectors(ids, doc_vecs, DOCS), namespace=namespace)
    index.upsert(vectors=to_pinecone_vectors(other_ids, other_vecs, OTHER_DOCS), namespace="other")
    print("Waiting until the upserted vectors are visible (eventual consistency)...")
    if not (wait_for_count(index, namespace, len(DOCS)) and wait_for_count(index, "other", len(OTHER_DOCS))):
        print("Warning: vectors not visible yet; results below may be incomplete.")

    # ---- 1. Same queries on both ---- #
    print(f"\n{'query':<38}{'same top-' + str(args.k):>10}  {'pinecone top1':<15}{'chroma top1':<15}{'pine ms':>9}{'chroma ms':>10}")
    print("-" * 99)
    for query, vec in zip(QUERIES, query_vecs):
        p_ids, p_scores = query_pinecone(index, vec, args.k, namespace)
        c_ids, c_scores = query_chroma(chroma, vec, args.k)
        p_ms = median_ms(lambda: query_pinecone(index, vec, args.k, namespace), args.repeats)
        c_ms = median_ms(lambda: query_chroma(chroma, vec, args.k), args.repeats)
        same = f"{overlap_at_k(p_ids, c_ids):.0%}"
        print(f"{query:<38}{same:>10}  {p_ids[0]} {p_scores[0]:.3f}  {c_ids[0]} {c_scores[0]:.3f}  {p_ms:>7.1f}{c_ms:>10.1f}")
    print("Latency is not a fair race: Chroma runs in your process, Pinecone includes a network round trip.")

    # ---- 2. Metadata filter on both ---- #
    vec = query_vecs[3]
    p_ids, _ = query_pinecone(index, vec, args.k, namespace, {"topic": {"$eq": "errors"}})
    c_ids, _ = query_chroma(chroma, vec, args.k, {"topic": "errors"})
    print(f"\nFilter topic=errors -> Pinecone {p_ids} | Chroma {c_ids}")

    # ---- 3. Namespaces ---- #
    print("\nNamespaces (same query vector, different namespace):")
    for ns in (namespace, "other"):
        found, _ = query_pinecone(index, query_vecs[2], 2, ns)
        print(f"  namespace '{ns}': {found}")
    print("Namespace stats:", {ns: v.get("vector_count") for ns, v in stats_dict(index).get("namespaces", {}).items()})

    # ---- 4. Cleanup ---- #
    if args.keep_index:
        print(f"\nIndex '{args.index}' kept (free plans limit how many indexes you can have).")
    else:
        pc.delete_index(args.index)
        print(f"\nDeleted index '{args.index}'.")


if __name__ == "__main__":
    main()
