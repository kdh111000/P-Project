"""Import Original_Data/{Bottoms,Tops,Outers} images into Supabase.

Run from the project root: python -m database.insertion_supabase --help
Uses a server-side admin key; never expose this importer as a public endpoint.

Created user
1111@1111.com
123456
"""

import argparse
import hashlib
import json
import sys
import os

from pathlib import Path
from uuid import UUID, NAMESPACE_URL, uuid5
from PIL import Image
from dotenv import load_dotenv

# 프로젝트 루트의 .env 로드
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

CATEGORIES = {"Bottoms": "bottom", "Tops": "top", "Outers": "outer"}
FORMATS = {"JPEG": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"),
           "WEBP": ("webp", "image/webp")}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_ROOT = Path(__file__).resolve().parent / "Original_Data"
MAX_BYTES = 6 * 1024 * 1024


def build_record(path: Path, root: Path, user_id: str, bucket: str) -> dict:
    """Validate the actual image, then build stable metadata without remote writes."""
    relative = path.relative_to(root)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Symbolic links or paths outside the input directory are not supported")
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Image must be nonempty and at most 6 MiB")
    with Image.open(path) as image:
        fmt = image.format
        width, height = image.size
        image.verify()
    if fmt not in FORMATS:
        raise ValueError("Only JPEG, PNG and WebP images are supported")
    extension, mime = FORMATS[fmt]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Identity is owner + relative path. Renaming a file creates a new clothing item.
    item_id = str(uuid5(NAMESPACE_URL, f"ai-closet:{user_id}:{relative.as_posix()}"))
    return {
        "id": item_id,
        "user_id": user_id,
        "name": path.stem,
        "category": CATEGORIES[relative.parts[0]],
        "source_path": relative.as_posix(),
        "image_bucket": bucket,
        "image_path": f"{user_id}/{item_id}/{digest}.{extension}",
        "image_sha256": digest,
        "image_mime_type": mime,
        "image_width": width,
        "image_height": height,
    }


def upload_one(client, path: Path, record: dict) -> str:
    existing = client.table("clothes").select("id,image_sha256").eq(
        "id", record["id"]
    ).execute().data
    if existing:
        if existing[0]["image_sha256"] != record["image_sha256"]:
            raise ValueError("This source path was already imported with different content; update explicitly")
        return "SKIP"
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["image_sha256"]:
        raise ValueError("Image changed during import; retry")
    client.storage.from_(record["image_bucket"]).upload(
        path=record["image_path"], file=payload,
        file_options={"content-type": record["image_mime_type"], "upsert": "true"},
    )
    # Only insert: preserve any existing user-edited name, state, or storage location.
    client.table("clothes").upsert(
        record, on_conflict="id", ignore_duplicates=True
    ).execute()
    return "OK"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--user-id",type=UUID,default=os.getenv("IMPORT_USER_ID", "").strip() or None,
    help="옷 소유자 UUID. 생략하면 .env의 IMPORT_USER_ID 사용",)
    parser.add_argument("--bucket", default="clothes-images")
    parser.add_argument("--dry-run", action="store_true", help="Validate locally without any Supabase calls")
    args = parser.parse_args()
    if args.user_id is None:
        parser.error(
        "--user-id 또는 .env의 IMPORT_USER_ID를 설정해주세요.")

    root = args.root.resolve()
    missing = [name for name in CATEGORIES if not (root / name).is_dir()]
    if missing:
        parser.error(f"Missing input folders under {root}: {', '.join(missing)}")
    prepared, errors = [], 0
    for folder in CATEGORIES:
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                prepared.append((path, build_record(path, root, str(args.user_id), args.bucket)))
            except Exception as exc:
                errors += 1
                print(f"INVALID {path.name}: {exc}", file=sys.stderr)
    if not prepared:
        print("No valid JPEG/PNG/WebP images found.", file=sys.stderr)
        return 1
    if args.dry_run:
        for _, record in prepared:
            print(json.dumps(record, ensure_ascii=False))
        print(f"DRY RUN: valid={len(prepared)}, invalid={errors}; no remote changes")
        return int(errors > 0)
    
    from clients import get_supabase_client
    client = get_supabase_client()
    # Check both resources before uploading images.
    client.storage.get_bucket(args.bucket)
    client.table("clothes").select("id").limit(1).execute()
    counts = {"OK": 0, "SKIP": 0}
    for path, record in prepared:
        try:
            status = upload_one(client, path, record)
            counts[status] += 1
            print(f"{status} {record['source_path']}")
        except Exception as exc:
            errors += 1
            print(f"FAILED {record['source_path']}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"Imported={counts['OK']}, skipped={counts['SKIP']}, failed={errors}")
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
