"""Index analyzed Supabase descriptions in Chroma; never download images.

Copy beside clients.py in DataBase/Scripts, then run from the project root:
python -m DataBase.Scripts.insertion_chroma --dry-run
python -m DataBase.Scripts.insertion_chroma
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR.parent


def iter_rows(db, user_id):
    offset = 0
    while True:
        rows = (db.table("clothes").select(
            "id,user_id,category,description,image_sha256,analyzed_image_sha256"
        ).eq("user_id", user_id).eq("analysis_status", "completed")
            .order("id").range(offset, offset + 99).execute().data)
        if not rows:
            return
        yield from rows
        offset += len(rows)


def eligible(row):
    return (
        isinstance(row.get("description"), str)
        and bool(row["description"].strip())
        and bool(row.get("image_sha256"))
        and row["image_sha256"] == row.get("analyzed_image_sha256")
        and row.get("category") in {"top", "bottom", "outer"}
    )


def get_collection(client, name, model, dimensions):
    collection = client.get_or_create_collection(
        name=name, embedding_function=None,
        metadata={"embedding_model": model, "embedding_dimensions": dimensions,
                  "hnsw:space": "cosine"},
    )
    metadata = collection.metadata or {}
    if (metadata.get("embedding_model") != model
            or metadata.get("embedding_dimensions") != dimensions
            or metadata.get("hnsw:space") != "cosine"):
        raise ValueError("Collection configuration differs or is unknown. Use a NEW collection name; do not mix models.")
    return collection


def index_one(collection, ai, row, model, dimensions, force=False):
    description = row["description"].strip()
    metadata = {
        "user_id": str(row["user_id"]),
        "category": row["category"],
        "image_sha256": row["image_sha256"],
        "embedding_model": model,
        "embedding_dimensions": dimensions,
    }
    signature = json.dumps({"description": description, **metadata}, sort_keys=True, ensure_ascii=False)
    metadata["index_hash"] = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    existing = collection.get(ids=[str(row["id"])], include=["metadatas"])
    if existing["ids"]:
        previous = (existing.get("metadatas") or [{}])[0] or {}
        if previous.get("user_id") != str(row["user_id"]):
            raise ValueError("Existing Chroma ID belongs to a different or unknown user")
        if not force and previous.get("index_hash") == metadata["index_hash"]:
            return "skipped"
    response = ai.embeddings.create(
        model=model, input=description, dimensions=dimensions, encoding_format="float",
    )
    embedding = response.data[0].embedding
    if len(embedding) != dimensions:
        raise ValueError("Unexpected embedding dimensions")
    collection.upsert(
        ids=[str(row["id"])], documents=[description],
        embeddings=[embedding], metadatas=[metadata],
    )
    return "saved"


def main():
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=UUID, default=os.getenv("IMPORT_USER_ID") or None)
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "clothes_descriptions_v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--dimensions", type=int, default=os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))
    parser.add_argument("--limit", type=int, default=0, help="Inspect at most N eligible records; 0 means all")
    parser.add_argument("--force", action="store_true", help="Re-embed unchanged descriptions too")
    parser.add_argument("--dry-run", action="store_true", help="Read Supabase only; no Chroma or OpenAI calls")
    args = parser.parse_args()
    if args.user_id is None:
        parser.error("Set IMPORT_USER_ID or pass --user-id")
    if args.limit < 0 or args.dimensions < 1:
        parser.error("--limit must be >= 0; --dimensions must be positive")
    if not args.dry_run and not os.getenv("OPENAI_API_KEY", "").strip():
        parser.error("Set OPENAI_API_KEY in .env")
    if __package__:
        from .clients import get_chroma_client, get_supabase_client
    else:
        from clients import get_chroma_client, get_supabase_client
    db = get_supabase_client()
    print("Loading analyzed descriptions...")
    rows = list(iter_rows(db, str(args.user_id)))
    targets = [row for row in rows if eligible(row)]
    print(f"Completed={len(rows)}, eligible={len(targets)}, invalid_or_stale={len(rows)-len(targets)}")
    if args.limit:
        targets = targets[:args.limit]
    if not targets:
        print("No eligible descriptions. Complete image analysis first.")
        return 0
    if args.dry_run:
        for row in targets:
            print(f"CANDIDATE {row['id']} category={row['category']}")
        print(f"DRY RUN: {len(targets)} candidates; existing Chroma entries not checked")
        return 0
    collection = get_collection(get_chroma_client(), args.collection, args.model, args.dimensions)
    saved = skipped = failed = 0
    with OpenAI(timeout=60.0, max_retries=2) as ai:
        try:
            with tqdm(total=len(targets), desc="Indexing", unit="item", dynamic_ncols=True) as progress:
                for row in targets:
                    try:
                        status = index_one(collection, ai, row, args.model, args.dimensions, args.force)
                        if status == "saved":
                            saved += 1
                        else:
                            skipped += 1
                    except Exception as exc:
                        failed += 1
                        reason = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}; status={getattr(exc, 'status_code', None)}"
                        tqdm.write(f"FAILED {row['id']}: {reason}")
                    progress.set_postfix(saved=saved, skipped=skipped, failed=failed, refresh=False)
                    progress.update(1)
        except KeyboardInterrupt:
            print(f"\nInterrupted: saved={saved}, skipped={skipped}, failed={failed}. Rerun to resume.")
            return 130
    print(f"DONE: saved={saved}, skipped={skipped}, failed={failed}; collection={args.collection}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
