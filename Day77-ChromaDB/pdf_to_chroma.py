"""Day 77 - Store PDF chunks in ChromaDB with metadata (source, page) and search with filters.

    python pdf_to_chroma.py index --demo
    python pdf_to_chroma.py index --pdf manual.pdf guide.pdf
    python pdf_to_chroma.py search "how do I reset it" --source manual.pdf --page-min 2
    python pdf_to_chroma.py search "warranty" --contains two
    python pdf_to_chroma.py stats
"""
import argparse
import os
import re
import sys
from collections import Counter

# Tiny built-in "PDFs" (one string per page) so you can try everything without a PDF.
DEMO_PDFS = {
    "router_manual.pdf": [
        "The Nimbus X1 is a dual-band Wi-Fi router for small homes. It supports speeds up to 1200 Mbps. "
        "To set up the router, plug the Ethernet cable from your modem into the blue WAN port and wait until the light turns green.",
        "To restore factory settings, hold the reset button for ten seconds until all lights blink once. "
        "Firmware updates are released every three months. Open the Maintenance tab and click Check Now.",
        "A solid red light means there is no internet connection. Restart the modem first, then the router. "
        "The warranty covers manufacturing defects for two years. Water damage is not covered.",
    ],
    "billing_guide.pdf": [
        "Invoices are emailed on the first day of every month. You can download any invoice as a PDF from the billing page.",
        "To change your subscription plan, open Billing Settings and choose a new plan. The change applies from the next billing cycle. "
        "If a payment fails, we retry three times over seven days.",
        "Refunds are issued within 14 days of purchase for damaged or defective items. "
        "Error ERR_4021 means the payment token has expired. Start a new checkout session.",
    ],
}


# ------------------------------------------------------------ pure helpers --

def chunk_text(text, size=300):
    """Split into sentences, then merge sentences into chunks of at most `size` characters."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    chunks, current = [], ""
    for sentence in filter(None, sentences):
        while len(sentence) > size:  # a single huge sentence: hard cut
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:size])
            sentence = sentence[size:]
        candidate = f"{current} {sentence}" if current else sentence
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def build_records(pages_by_source, size=300):
    """Turn {source: [page texts]} into parallel lists of ids, documents, metadatas."""
    ids, documents, metadatas = [], [], []
    for source, pages in pages_by_source.items():
        for page_number, page_text in enumerate(pages, start=1):  # pages are 1-based
            for chunk_number, chunk in enumerate(chunk_text(page_text, size)):
                ids.append(f"{source}-p{page_number}-c{chunk_number}")  # stable id -> re-indexing is safe
                documents.append(chunk)
                metadatas.append({"source": source, "page": page_number, "chunk": chunk_number})
    return ids, documents, metadatas


def build_where(sources=None, page_min=None, page_max=None):
    """Chroma metadata filter. Several conditions must be combined with $and."""
    conditions = []
    if sources:
        conditions.append({"source": sources[0]} if len(sources) == 1 else {"source": {"$in": list(sources)}})
    if page_min is not None:
        conditions.append({"page": {"$gte": page_min}})
    if page_max is not None:
        conditions.append({"page": {"$lte": page_max}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def count_by_source(metadatas):
    return Counter(meta["source"] for meta in metadatas)


# ----------------------------------------------------------------- Chroma --

def make_embedding_function(name):
    from chromadb.utils import embedding_functions

    if name == "default":  # Chroma's built-in small local model
        return embedding_functions.DefaultEmbeddingFunction()
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=name)


def open_collection(client, name, embedder, create):
    """
    Always reopen a collection with the SAME embedding function it was built with,
    otherwise query vectors live in a different space and results are meaningless.
    The embedder name is saved in the collection metadata for this reason.
    """
    try:
        existing = client.get_collection(name)
    except Exception:
        existing = None

    if existing is not None:
        stored = (existing.metadata or {}).get("embedder", "default")
        if embedder and embedder != stored:
            sys.exit(f"Collection '{name}' was built with embedder '{stored}', not '{embedder}'. "
                     "Delete the database folder or use the same embedder.")
        return client.get_collection(name, embedding_function=make_embedding_function(stored))

    if not create:
        sys.exit(f"No collection '{name}' yet. Run the 'index' command first.")
    embedder = embedder or "default"
    return client.create_collection(
        name, embedding_function=make_embedding_function(embedder),
        metadata={"hnsw:space": "cosine", "embedder": embedder})


def read_pdf_pages(path):
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(path).pages]


def cmd_index(args, client):
    if args.demo:
        pages_by_source = DEMO_PDFS
    elif args.pdf:
        pages_by_source = {os.path.basename(path): read_pdf_pages(path) for path in args.pdf}
    else:
        sys.exit("Give --demo or --pdf file.pdf [more.pdf ...]")

    collection = open_collection(client, args.collection, args.embedder, create=True)
    ids, documents, metadatas = build_records(pages_by_source, args.chunk_size)
    if not ids:
        sys.exit("No text found (scanned PDFs need OCR).")
    for start in range(0, len(ids), 500):  # upsert in batches: safe to run again
        end = start + 500
        collection.upsert(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])
    print(f"Indexed {len(ids)} chunks from {len(pages_by_source)} file(s). Collection now has {collection.count()}.")


def cmd_search(args, client):
    collection = open_collection(client, args.collection, None, create=False)
    where = build_where(args.source, args.page_min, args.page_max)
    where_document = {"$contains": args.contains} if args.contains else None
    result = collection.query(query_texts=[args.query], n_results=args.k,
                              where=where, where_document=where_document)

    print(f'Query: "{args.query}"  filter: {where}  contains: {args.contains}')
    if not result["ids"][0]:
        print("  (no results: the filter matched nothing)")
    for doc, distance, meta in zip(result["documents"][0], result["distances"][0], result["metadatas"][0]):
        print(f"  sim={1 - distance:+.3f}  {meta['source']} p.{meta['page']}: {doc[:90]}")


def cmd_stats(args, client):
    collection = open_collection(client, args.collection, None, create=False)
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    print(f"Collection '{args.collection}': {len(metadatas)} chunks, "
          f"embedder: {(collection.metadata or {}).get('embedder', 'default')}")
    for source, count in sorted(count_by_source(metadatas).items()):
        pages = sorted({m["page"] for m in metadatas if m["source"] == source})
        print(f"  {source}: {count} chunks, pages {pages[0]}-{pages[-1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="chroma_db", help="Database folder (default: chroma_db)")
    parser.add_argument("--collection", default="pdf_docs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="Add PDFs (or demo data) to the collection")
    p.add_argument("--pdf", nargs="+")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--chunk-size", type=int, default=300)
    p.add_argument("--embedder", default=None,
                   help='"default" (built in) or a sentence-transformers model name, e.g. all-MiniLM-L6-v2')

    p = sub.add_parser("search", help="Semantic search with optional filters")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--source", nargs="+", help="Only these files")
    p.add_argument("--page-min", type=int)
    p.add_argument("--page-max", type=int)
    p.add_argument("--contains", help="Only chunks whose text contains this string")

    sub.add_parser("stats", help="Show what is stored")

    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=args.path)  # data survives between runs
    {"index": cmd_index, "search": cmd_search, "stats": cmd_stats}[args.command](args, client)


if __name__ == "__main__":
    main()
