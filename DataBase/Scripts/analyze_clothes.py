"""Analyze Supabase clothing images. Copy beside clients.py in DataBase/Scripts.

python DataBase/Scripts/analyze_clothes.py --dry-run
python DataBase/Scripts/analyze_clothes.py --limit 1
"""
import argparse
import base64
import hashlib
import io
import os

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict
from DataBase.prompts import CLOTHES_ANALYSIS_PROMPT

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR.parent


class ClothingAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_single_clothing_item: bool
    clothing_type: str | None
    colors: list[str]
    pattern: str | None
    sleeve_length: str | None
    garment_length: str | None
    apparent_fit: str | None
    visible_details: list[str]
    style_suggestions: list[str]
    description: str


def needs_analysis(row: dict, force: bool = False) -> bool:
    return force or not (
        row.get("analysis_status") == "completed"
        and row.get("analyzed_image_sha256") == row.get("image_sha256")
        and bool(row.get("description"))
    )


def analyze_image(ai, payload: bytes, model: str) -> ClothingAnalysis:
    if not 0 < len(payload) <= 6 * 1024 * 1024:
        raise ValueError("Image must be between 1 byte and 6 MiB")
    with Image.open(io.BytesIO(payload)) as image:
        image_format = image.format
        if image_format is None:
            raise ValueError("Could not determine image format")
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(image_format)
        image.verify()
    if mime is None:
        raise ValueError("Unsupported image format")
    encoded = base64.b64encode(payload).decode("ascii")
    response = ai.responses.parse(
        model=model,
        instructions=CLOTHES_ANALYSIS_PROMPT,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": "사진의 의류 한 벌을 분석해주세요."},
            {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}", "detail": "auto"},
        ]}],
        text_format=ClothingAnalysis,
        max_output_tokens=2000,
        store=False,
    )
    result = response.output_parsed
    if response.status != "completed" or result is None:
        raise ValueError("Model refused or returned incomplete/unparseable output")
    if not result.is_single_clothing_item or not result.description.strip():
        raise ValueError("No usable description of a single clothing item")
    return result


def guarded_update(db, row: dict, values: dict):
    # Do not attach an old image's result to a replaced image.
    return (db.table("clothes").update(values)
            .eq("id", row["id"]).eq("user_id", row["user_id"])
            .eq("image_sha256", row["image_sha256"])
            .eq("image_path", row["image_path"])
            .eq("image_bucket", row["image_bucket"]).execute())


def process_one(db, ai, row: dict, model: str) -> None:
    payload = db.storage.from_(row["image_bucket"]).download(row["image_path"])
    if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
        raise ValueError("Storage image hash differs from the clothes record")
    result = analyze_image(ai, payload, model)
    saved = guarded_update(db, row, {
        "description": result.description.strip(),
        "color": ", ".join(result.colors) or None,
        "pattern": result.pattern,
        "attributes": result.model_dump(exclude={"description"}),
        "analysis_status": "completed",
        "analyzed_image_sha256": row["image_sha256"],
        "analysis_model": model,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "analysis_error": None,
    })
    if not saved.data:
        raise RuntimeError("Record changed or update was denied; no result saved")


def iter_rows(db, user_id: str):
    offset = 0
    while True:
        rows = (db.table("clothes").select(
            "id,user_id,image_bucket,image_path,image_sha256,description,"
            "analysis_status,analyzed_image_sha256,attributes,analysis_model,analyzed_at,analysis_error"
        ).eq("user_id", user_id).order("id").range(offset, offset + 99).execute().data)
        if not rows:
            return
        yield from rows
        offset += len(rows)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=UUID, default=os.getenv("IMPORT_USER_ID") or None)
    parser.add_argument("--model", default=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum attempts; 0 means all")
    parser.add_argument("--dry-run", action="store_true", help="Read DB only; no downloads, AI calls or writes")
    parser.add_argument("--force", action="store_true", help="Reanalyze completed items too (API charges apply)")
    args = parser.parse_args()
    if args.user_id is None:
        parser.error("Set IMPORT_USER_ID in .env or supply --user-id")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if not args.dry_run and not os.getenv("OPENAI_API_KEY", "").strip():
        parser.error("Set OPENAI_API_KEY in the project .env")
    if __package__:
        from .clients import get_supabase_client
    else:
        from clients import get_supabase_client
    db = get_supabase_client()
    print("Loading clothing records...")
    rows = list(iter_rows(db, str(args.user_id)))
    targets = [row for row in rows if needs_analysis(row, args.force)]

    skipped = len(rows) - len(targets)
    deferred = max(0, len(targets) - args.limit) if args.limit else 0

    if args.limit:
        targets = targets[:args.limit]

    print(
        f"Total={len(rows)}, targets={len(targets)}, "
        f"already_done={skipped}, deferred={deferred}"
    )

    if not targets:
        print("No images need analysis.")
        return 0

    ai = None if args.dry_run else OpenAI(timeout=90.0, max_retries=2)
    succeeded = failed = processed = 0

    try:
        with tqdm(
            total=len(targets),
            desc="Preview" if args.dry_run else "Analyzing",
            unit="image",
            dynamic_ncols=True,
            mininterval=0.2,
        ) as progress:
            for row in targets:
                progress.set_postfix(
                    ok=succeeded,
                    failed=failed,
                    item=row["id"][:8],
                )

                if args.dry_run:
                    tqdm.write(f"WOULD ANALYZE {row['id']}")
                else:
                    try:
                        process_one(db, ai, row, args.model)
                        succeeded += 1
                    except Exception as exc:
                        failed += 1
                        # 변경: 코드에서 발생시킨 ValueError의 상세 메시지 표시
                        if isinstance(exc, ValueError):
                            reason = f"ValueError: {exc}"
                        else:
                            reason = (
                                f"{type(exc).__name__}; "
                                f"status={getattr(exc, 'status_code', None)}"
                            )
                        tqdm.write(f"FAILED {row['id']}: {reason}")

                        try:
                            values = {"analysis_error": reason}
                            if needs_analysis(row):
                                values["analysis_status"] = "failed"
                            guarded_update(db, row, values)
                        except Exception as save_error:
                            tqdm.write(
                                "FAILED saving error status: "
                                f"{type(save_error).__name__}"
                            )

                processed += 1
                progress.set_postfix(
                    ok=succeeded, failed=failed, refresh=False
                )
                progress.update(1)

        print(
            f"{'DRY RUN' if args.dry_run else 'DONE'}: "
            f"processed={processed}, success={succeeded}, "
            f"skipped={skipped}, failed={failed}, deferred={deferred}"
        )
        return int(failed > 0)

    except KeyboardInterrupt:
        print(
            f"\nInterrupted: processed={processed}/{len(targets)}, "
            f"success={succeeded}, failed={failed}."
        )
        print(
            "Saved results remain in Supabase. "
            "Run again without --force to resume."
        )
        return 130

    finally:
        if ai is not None:
            ai.close()


if __name__ == "__main__":
    raise SystemExit(main())
