"""The door camera in a browser: pick any camera of the device viewing the page and any trained 26 on this PC,
then watch the boxes, the in/out moves and the registered / unregistered calls live (live.py does the work).
  python camweb.py [--port 8000] [--http]
https with a self-signed certificate, because browsers give cameras only to https pages (or localhost): the browser warns
once, choose 'advanced -> proceed'. The page needs the key printed at the start (camweb_key.txt), so the open port is not
anyone's CPU. Models: models/held.pt or else match/held.pt (in use) and every 26 in models/, with its note from models/목록.txt.
Ticking "학습용 장면 저장" keeps frames of moves seen with a still camera on this PC (collect.py); /review sorts them.
"녹화" keeps every frame received as doorcam_web/<time>/rec/<unix time>.jpg: feeding them back to live.run in name order
replays the session exactly (for scoring moves). Those are the frames the judge saw (960 wide, as fast as it judges); the camera's
own video (its full size and frame rate) comes too, in 5 s pieces while it runs: doorcam_web/<time>/raw.webm (or .mp4) + raw.json.
A phone stores it its own way up (held sideways, the video lies on its side, with no mark saying so), so raw.json "shape"
keeps, piece by piece, the screen's turn (screen.orientation.angle) and the picture's size as shown: [piece, angle, w, h]."""
import argparse, asyncio, fnmatch, ipaddress, json, queue, re, secrets, socket, tempfile, threading, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import collect
import live

ROOT = live.ROOT
MODELS, PAGE, REVIEW, KEY_FILE, CERT, OUT = ROOT / "models", ROOT / "camweb.html", ROOT / "camweb_review.html", ROOT / "camweb_key.txt", ROOT / "camweb_cert", ROOT / "doorcam_web"
NOT_26 = ("포즈", "match_", "held.")
MAX_BYTES = 15 * 1024 * 1024
MAX_RAW = 100 * 1024 * 1024  # one 5 s piece of the camera's own video
KEY = KEY_FILE.read_text().strip() if KEY_FILE.exists() else ""
if not KEY:
    KEY = secrets.token_urlsafe(12)
    KEY_FILE.write_text(KEY)
app = FastAPI()


def models():
    """{name shown: (path, note)}"""
    notes = []
    if (MODELS / "목록.txt").exists():
        for l in (MODELS / "목록.txt").read_text(encoding="utf-8").splitlines():
            if len(p := re.split(r"\s{2,}", l.strip(), maxsplit=1)) == 2:
                notes.append(([n.strip() for n in p[0].split(",")], p[1]))
    note = lambda name: next((t for ns, t in notes if any(fnmatch.fnmatch(name, n) for n in ns)), "")
    out = {f"현재 사용 ({live.HELD.relative_to(ROOT).as_posix()})": (live.HELD, note(live.HELD.name) if live.HELD.parent == MODELS else "6차 (공개판)")}
    for p in sorted(MODELS.glob("*.pt")):
        if not p.name.startswith(NOT_26):
            out[p.stem] = (p, note(p.name))
    return out


def check(key):
    if not secrets.compare_digest(key or "", KEY):
        raise HTTPException(403, "주소의 key가 맞지 않습니다.")


@app.get("/")
def page():
    return FileResponse(PAGE)


@app.get("/api/models")
def api_models(key: str = ""):
    check(key)
    return {"models": [{"name": k, "note": n} for k, (_, n) in models().items()], "registered": list(live.garments()),
            "conf": live.CONF, "pending": len(collect.pending())}


@app.get("/review")
def review():
    return FileResponse(REVIEW)


@app.get("/api/collect")
def api_collect(key: str = ""):
    check(key)
    return {"items": collect.pending(), "approved": collect.approved()}


@app.get("/api/collect/item")
def api_collect_item(key: str = "", id: str = ""):
    check(key)
    try:
        return collect.item(id)[1]
    except KeyError:
        raise HTTPException(404, "없는 장면입니다.")


@app.get("/api/collect/img")
def api_collect_img(key: str = "", id: str = ""):
    check(key)
    try:
        return FileResponse(collect.item(id)[0])
    except KeyError:
        raise HTTPException(404, "없는 장면입니다.")


@app.post("/api/collect/judge")
def api_collect_judge(key: str = Form(""), id: str = Form(...), verdict: str = Form(...), boxes: str = Form("")):
    check(key)
    try:
        collect.judge(id, verdict, json.loads(boxes) if boxes else None)
    except (KeyError, ValueError, TypeError, IndexError):
        raise HTTPException(400, "판정할 수 없습니다.")
    return {"left": len(collect.pending()), "approved": collect.approved()}



@app.get("/api/events")
def api_events(key: str = "", since: str = "", limit: int = 500):
    """the camera's calls (live.EVENTS), oldest first: only those after since (a "time" from an earlier answer), the newest limit"""
    check(key)
    lines = live.EVENTS.read_text(encoding="utf-8").splitlines() if live.EVENTS.exists() else []  # ponytail: whole file, fine to ~100k calls
    ev = [e for e in map(json.loads, filter(str.strip, lines)) if e["time"] > since]
    return {"events": ev[-max(1, min(limit, 5000)):]}


@app.get("/api/crop")
def api_crop(key: str = "", path: str = ""):
    """the crops a call was made on (an event's "crop")"""
    check(key)
    f = (live.ROOT / path).resolve()
    if f.suffix != ".jpg" or not f.is_relative_to(OUT.resolve()) or not f.is_file():
        raise HTTPException(404, "없는 사진입니다.")
    return FileResponse(f)


@app.post("/api/register")
async def api_register(key: str = Form(""), name: str = Form(...), front: UploadFile = File(...), back: UploadFile = File(...),
                       side: UploadFile = File(...)):
    check(key)
    with tempfile.TemporaryDirectory() as d:
        files = []
        for v, f in zip(live.VIEWS, (front, back, side)):
            b = await f.read()
            if len(b) > MAX_BYTES or cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) is None:
                raise HTTPException(400, f"사진을 읽을 수 없습니다: {v}")
            (p := Path(d) / f"{v}.jpg").write_bytes(b)
            files.append(str(p))
        try:
            await asyncio.to_thread(live.add, name.strip(), files)
        except SystemExit as e:  # live.add says what is wrong this way
            raise HTTPException(400, str(e))
    return {"registered": list(live.garments())}


@app.post("/api/raw")
async def api_raw(key: str = Form(""), folder: str = Form(...), n: int = Form(...), type: str = Form(...), t0: float = Form(...),
                  sent: float = Form(...), angle: int = Form(0), w: int = Form(0), h: int = Form(0), chunk: UploadFile = File(...)):
    """piece n of the camera's own video of one session, appended in order (the page sends them one after another)"""
    check(key)
    d, ext = OUT / folder, "mp4" if type.startswith("video/mp4") else "webm"
    if not re.fullmatch(r"\d{8}-\d{6}-\d{6}", folder) or not (d / "rec").is_dir():
        raise HTTPException(400, "없는 녹화입니다.")
    meta = json.loads((d / "raw.json").read_text()) if (d / "raw.json").exists() else {"pieces": 0}  # one session at a time
    if n < meta["pieces"]:
        return meta  # sent twice (a retry after a lost answer)
    if n != meta["pieces"] or (b := await chunk.read(MAX_RAW + 1)) is None or len(b) > MAX_RAW:
        raise HTTPException(400, f"조각 순서가 맞지 않습니다: {n} (기대 {meta['pieces']})")
    with open(d / f"raw.{ext}", "ab") as f:
        f.write(b)
    if n == 0:  # page and PC clocks differ: rec/<unix time>.jpg lines up with the video through server_got - sent
        meta.update(type=type, file=f"raw.{ext}", page_start_ms=t0, sent_ms=sent, server_got=time.time())
    if not meta.get("shape") or meta["shape"][-1][1:] != [angle, w, h]:  # the phone turned (or the first piece)
        meta.setdefault("shape", []).append([n, angle, w, h])
    meta["pieces"], meta["bytes"] = n + 1, meta.get("bytes", 0) + len(b)
    (d / "raw.json").write_text(json.dumps(meta))
    return meta



@app.websocket("/ws")
async def ws(sock: WebSocket, key: str = "", model: str = "", side: str = "right", size: int = 640, keep: int = 0, rec: int = 0):
    """the page sends JPEG frames, one at a time; each answer: boxes, moves so far, whether a move is being judged"""
    m = models().get(model)
    if not secrets.compare_digest(key, KEY) or m is None or side not in ("left", "right") or size not in (480, 640, 960):
        await sock.close(1008)
        return
    await sock.accept()
    frames, answers, moves = queue.Queue(), queue.Queue(), queue.Queue()
    ses = collect.Session(model) if keep else None
    out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    rec_dir, n_rec = out / "rec", 0
    if rec:
        rec_dir.mkdir(parents=True)
        await sock.send_text(json.dumps({"folder": out.name}))  # where the camera's own video goes (/api/raw)

    def feed():
        while (f := frames.get()) is not None:
            yield f

    def watch():
        try:
            live.run(feed(), side, size=size, window=False, out=out, held=m[0],
                     on_frame=lambda d, l, busy: answers.put({"dets": d, "head": l[0].split(" · q")[0], "moves": l[1:], "busy": busy}),
                     on_move=lambda *m: moves.put(m))
        except BaseException as e:  # SystemExit: a model file missing
            answers.put({"error": str(e) or type(e).__name__})

    threading.Thread(target=watch, daemon=True).start()
    try:
        while True:
            b = await sock.receive_bytes()
            bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            t = time.time()
            if rec:
                (rec_dir / f"{t:.3f}.jpg").write_bytes(b)
                n_rec += 1
            frames.put((t, bgr))
            try:
                a = await asyncio.to_thread(answers.get, True, 120)  # the first frame waits for the models to open
            except queue.Empty:
                a = {"error": "응답이 없습니다."}
            if ses:
                a["kept"] = ses.n
            if rec:
                a["rec"] = n_rec
            await sock.send_text(json.dumps({**a, "ms": round((time.time() - t) * 1000)}, ensure_ascii=False))
            if "error" in a:
                break
            if ses:  # after the answer: the page is busy with the next frame meanwhile
                await asyncio.to_thread(ses.frame, t, b, bgr, a["dets"])
                while not moves.empty():
                    ses.move(*moves.get())
    except WebSocketDisconnect:
        pass
    finally:
        frames.put(None)  # live.run judges what is left, then ends


def cert():
    """a self-signed certificate for this PC's names and addresses, made once"""
    key, crt = CERT / "key.pem", CERT / "cert.pem"
    if crt.exists():
        return key, crt
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    k = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "closet camweb")])
    alt = [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())] + [x509.IPAddress(ipaddress.ip_address(a)) for a in {"127.0.0.1", *ips()}]
    now = datetime.now(timezone.utc)
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=825))
         .add_extension(x509.SubjectAlternativeName(alt), False).sign(k, hashes.SHA256()))
    CERT.mkdir(exist_ok=True)
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    crt.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    return key, crt


def ips():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.0.1", 9))  # no packet is sent: only picks the address of the way out
        return [s.getsockname()[0]]
    except OSError:
        return []
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--http", action="store_true", help="no https: cameras then work only at http://localhost")
    a = ap.parse_args()
    s = "http" if a.http else "https"
    for h in ["localhost"] + ips():
        print(f"{s}://{h}:{a.port}/?key={KEY}", flush=True)
    print(f"포트포워딩했다면 다른 곳에서: {s}://<공인 IP>:{a.port}/?key={KEY}", flush=True)
    k, c = (None, None) if a.http else cert()
    uvicorn.run(app, host="0.0.0.0", port=a.port, ssl_keyfile=k and str(k), ssl_certfile=c and str(c), log_level="warning")


if __name__ == "__main__":
    main()
