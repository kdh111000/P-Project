"""The team's Fitmate DB (Supabase, table clothes) <-> the door camera, over Supabase's REST API.
Set in the environment (never in a file here):  SUPABASE_URL, SUPABASE_KEY (a server key: the camera writes), FITMATE_USER_ID
Unset -> off: everything stays on this PC exactly as before.
  pull(): the user's clothes rows + their photo -> db_cache/<id>/front.<ext>, used as registered garments (live.garments).
          The DB keeps one photo per garment: back.* / side.* put in the same folder by hand stay local and are used too.
  push(e): after a move of a DB garment, the columns the table has get it; what it has no column for -> db_local.jsonl.
Unregistered / unsure garments have no row: they stay in the local log (track.jsonl) only."""
import json, os, shutil
from datetime import datetime
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
CACHE, LOCAL = ROOT / "db_cache", ROOT / "db_local.jsonl"
TABLE = "clothes"
IN_CLOSET = True  # is_available = the garment is really in the closet (team, 11-29)
TIMEOUT = 15


def env():
    return os.environ.get("SUPABASE_URL", "").rstrip("/"), os.environ.get("SUPABASE_KEY", ""), os.environ.get("FITMATE_USER_ID", "")


def on():
    return all(env())


def _get(path, **kw):
    url, key, _ = env()
    r = requests.request(kw.pop("method", "GET"), f"{url}{path}", timeout=TIMEOUT,
                         headers={"apikey": key, "Authorization": f"Bearer {key}", **kw.pop("headers", {})}, **kw)
    r.raise_for_status()
    return r


def ids():
    """{garment key shown: row id} of what the last pull kept"""
    f = CACHE / "ids.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def columns():
    f = CACHE / "columns.json"
    return set(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else set()


def garments(taken=()):
    """{key: folder} from the cache (no network); taken: names already used on this PC"""
    return {k: CACHE / i for k, i in ids().items() if k not in taken and (CACHE / i).is_dir()}


def pull(taken=()):
    """refresh the cache from the DB -> number of garments; raises on network / key trouble (the old cache stays)"""
    _, _, user = env()
    rows = _get(f"/rest/v1/{TABLE}", params={"select": "*", "user_id": f"eq.{user}"}).json()
    CACHE.mkdir(exist_ok=True)
    keys, cols = {}, set().union(*(r.keys() for r in rows)) if rows else columns()
    for r in rows:
        d, meta = CACHE / r["id"], CACHE / r["id"] / "row.json"
        old = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        if old.get("image_sha256") != r.get("image_sha256") or not list(d.glob("front.*")):
            img = _get(f"/storage/v1/object/{r['image_bucket']}/{r['image_path']}").content
            d.mkdir(exist_ok=True)
            for p in d.glob("front.*"):
                p.unlink()
            (d / ("front" + Path(r["image_path"]).suffix.lower())).write_bytes(img)
        meta.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        name = r.get("name") or r["id"][:8]
        keys[name if name not in keys and name not in taken else f"{name}·{r['id'][:4]}"] = r["id"]
    for d in CACHE.iterdir():  # gone from the DB: its photos go too (hand-added back/side with it)
        if d.is_dir() and d.name not in keys.values():
            shutil.rmtree(d)
    (CACHE / "ids.json").write_text(json.dumps(keys, ensure_ascii=False, indent=1), encoding="utf-8")
    (CACHE / "columns.json").write_text(json.dumps(sorted(cols)), encoding="utf-8")
    return len(keys)


def push(e):
    """e: a move from track.Track.move. -> the columns written to the DB ([] if it is not a DB garment or the DB is off)"""
    i = ids().get(e["item"])
    if not (on() and i):
        return []
    want = {"is_available": (e["action"] == "in") == IN_CLOSET,
            "location_verified_at": datetime.fromtimestamp(e["end"]).astimezone().isoformat(),
            "last_action": e["action"], "last_move_started_at": datetime.fromtimestamp(e["start"]).astimezone().isoformat(),
            "match_p": e["p"], "move_crop": e.get("crop")}
    cols = columns()
    db = {k: v for k, v in want.items() if k in cols}
    rest = {k: v for k, v in want.items() if k not in cols}
    if db:
        _get(f"/rest/v1/{TABLE}", method="PATCH", params={"id": f"eq.{i}"}, json=db, headers={"Prefer": "return=minimal"})
    if rest:
        with LOCAL.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": i, "item": e["item"], **rest}, ensure_ascii=False) + "\n")
    return sorted(db)
