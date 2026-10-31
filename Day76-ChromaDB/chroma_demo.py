"""Day 76 - ChromaDB: create a collection, add documents, run queries."""
import argparse

import chromadb
import numpy as np
from chromadb.utils import embedding_functions

COLLECTION = "support_docs"

# (document, topic)
DOCS = [
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


def exact_top_k(vectors, query, k):
    """Brute-force cosine search over ALL vectors: the exact answer ANN tries to approximate."""
    vectors = np.asarray(vectors, dtype=float)
    query = np.asarray(query, dtype=float)
    scores = (vectors @ query) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
    return [int(i) for i in np.argsort(-scores)[:k]]


def show(title, result):
    print(f"\n{title}")
    for doc_id, doc, distance, meta in zip(result["ids"][0], result["documents"][0],
                                           result["distances"][0], result["metadatas"][0]):
        # With the cosine space, Chroma's distance = 1 - cosine similarity.
        print(f"  {doc_id:<7} sim={1 - distance:+.3f}  [{meta['topic']}] {doc[:55]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="chroma_db", help="Folder where the database is stored")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the saved collection and skip re-adding (shows persistence)")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.path)
    embed = embedding_functions.DefaultEmbeddingFunction()  # small local model, downloaded on first use

    if not args.keep:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass  # did not exist yet

    # Chroma defaults to L2 distance; ask for cosine so scores match Day 71-75.
    collection = client.get_or_create_collection(
        COLLECTION, embedding_function=embed, metadata={"hnsw:space": "cosine"})

    if collection.count() == 0:
        collection.add(
            ids=[f"doc-{i}" for i in range(len(DOCS))],
            documents=[text for text, _ in DOCS],
            metadatas=[{"topic": topic} for _, topic in DOCS],
        )
        print(f"Added {collection.count()} documents.")
    else:
        print(f"Loaded existing collection with {collection.count()} documents.")

    # 1. Basic query: Chroma embeds the query text with the same model.
    query = "I can't log into my account"
    show(f'Query: "{query}"', collection.query(query_texts=[query], n_results=3))

    # 2. Query + metadata filter
    query = "something went wrong"
    show(f'Query: "{query}"  (only topic = errors)',
         collection.query(query_texts=[query], n_results=3, where={"topic": "errors"}))

    # 3. ANN result vs. exact brute-force search over the same vectors
    stored = collection.get(include=["embeddings"])
    ids = list(stored["ids"])
    vectors = np.asarray(stored["embeddings"])
    query = "how long until my package arrives"
    ann_ids = collection.query(query_texts=[query], n_results=3)["ids"][0]
    query_vector = np.asarray(embed([query])[0])
    exact_ids = [ids[i] for i in exact_top_k(vectors, query_vector, 3)]
    print(f"\nANN (HNSW) top 3 : {ann_ids}\nExact top 3      : {exact_ids}\nSame result: {ann_ids == exact_ids}")

    # 4. Get / update / delete
    print("\nget doc-3:", collection.get(ids=["doc-3"])["documents"][0][:60])
    collection.upsert(ids=["doc-4"], documents=["Our store is open every day from 8 AM to 8 PM."],
                      metadatas=[{"topic": "store"}])
    print("after upsert, doc-4:", collection.get(ids=["doc-4"])["documents"][0])
    collection.delete(ids=["doc-9"])
    print("count after delete:", collection.count())

    print(f"\nData is saved in ./{args.path}. Run again with --keep to reuse it "
          "(note: --keep skips re-adding, so the upsert/delete above stay applied).")


if __name__ == "__main__":
    main()
