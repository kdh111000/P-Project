"""Closet door camera: an Orbbec Astra on top of the closet sees the user only while the hinged door is open.
Each door opening is one session, recorded while it happens (colour/NNNNN.jpg + depth/NNNNN.png at 10 fps, plus the first and
last frames with someone in front); a separate worker (--worker, the backend) later compares them with the user's garments
to log what was taken out and what was put back. The depth laser only runs while the colour image is bright (door open):
Astra got too hot to hold after 30 minutes of continuous depth.
Camera: Astra (USB 2BC5:0404) through OpenNI2 (`pip install openni` + Orbbec's OpenNI2 SDK folder, OPENNI2_DIR).
Someone = pixels clearly closer than the empty scene (doorcam/background.png), so it works wherever the camera points;
the empty scene is learned from the first second if missing, or again with `b` in the live window.
run: python doorcam.py --closet <closet id> [--show]   (camera: records, prints stats every second)
     python doorcam.py --worker                         (backend: analyses recorded sessions as they appear)"""
import argparse, collections, json, os, shutil, signal, time
from pathlib import Path
import cv2
import numpy as np

LOG = Path(__file__).resolve().parent / "doorcam"
BG = LOG / "background.png"
STOP = LOG / "stop"  # created by --stop; the camera loop exits cleanly when it sees it
OPENNI2_DIR = os.environ.get("OPENNI2_DIR", str(Path.home() / "Downloads/Orbbec/OpenNI2/sdk/libs"))
# ponytail: guessed for an Astra (depth range ~0.6-8 m); calibrate on the mounted camera
DARK = 35                   # door shut: it blocks the light; mean grey of the colour frame (a hand over the lens: ~10)
CLOSER_MM = 150             # this much in front of the empty scene = someone (Astra noise is a few cm at 2 m)
PERSON_FRAC = 0.05          # share of the frame they must fill
SHUT_FRAMES = 5             # a hand passing over the lens is not a closed door
STEP = 3                    # record every 3rd frame: 10 fps is enough for the backend and a third of the disk
IDLE_S = 10                 # dark this long (door shut): colour stream off too, only a quick look every POLL_S
POLL_S = 1.0                # ponytail: up to ~1 s late when the door opens; shorten if the first moment matters
WAKE_FRAMES = 5             # frames read after a restart before judging brightness (auto exposure)
JPEG_QUALITY = 80           # recorded colour frames (OpenCV default 95 gave ~135 KB per 640x480 frame)
KEEP_DAYS = 3               # raw recorded frames are deleted after this (the judged shots stay)
MAX_RAW_GB = 5.0            # ...or sooner, oldest first, when all raw frames together exceed this


def door_open(depth, bgr):
    """brightness only: the shut door blocks the light; depth coverage varies too much with where the camera points
    (0.16 valid on a bright, open scene)"""
    return bgr.mean() > DARK


def closer(depth, bg):
    """someone's pixels; ponytail: closer than the Astra's ~0.6 m minimum reads 0 and is missed"""
    return (depth > 0) & (bg > 0) & (bg.astype(np.int32) - depth > CLOSER_MM)


def person(depth, bg):
    return closer(depth, bg).mean() > PERSON_FRAC


def background(frames, n=30):
    """empty scene = per-pixel median of n frames (0 where the camera saw nothing)"""
    return np.median([next(frames)[2] for _ in range(n)], axis=0).astype(np.uint16)


def sessions(frames, bg, tape=None):
    """frames: (unix time, bgr, depth mm uint16) -> yields (opened, closed, before, after) per door opening;
    before/after = (bgr, depth) of the first/last frame with someone in front of the empty scene bg, None if nobody came.
    tape (a Tape) records every door-open frame while it happens."""
    opened = before = after = None
    shut = 0
    for t, bgr, depth in frames:
        if door_open(depth, bgr):
            shut = 0
            if opened is None:
                opened = t
                tape and tape.start(t)
            tape and tape.add(bgr, depth)
            if person(depth, bg):
                before = before or (bgr, depth)
                after = (bgr, depth)
        elif opened is not None:
            shut += 1
            if shut >= SHUT_FRAMES:
                tape and tape.stop()
                yield opened, t, before, after
                opened = before = after = None
    if opened is not None:
        tape and tape.stop()
        yield opened, t, before, after


def folder(t, out=LOG):
    return out / time.strftime("%Y%m%d-%H%M%S", time.localtime(t))


class Tape:
    """one door opening written to disk while it happens: colour/00000.jpg + depth/00000.png (16-bit mm), every
    STEP-th frame. Single files, not a video: an mp4 cut off by a killed process had no index and could not be read."""
    def __init__(self, out=LOG):
        self.out = out

    def start(self, t):
        self.d, self.n = folder(t, self.out), 0
        for sub in ("colour", "depth"):
            (self.d / sub).mkdir(parents=True, exist_ok=True)

    def add(self, bgr, depth):
        if self.n % STEP == 0:
            cv2.imwrite(str(self.d / "colour" / f"{self.n // STEP:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cv2.imwrite(str(self.d / "depth" / f"{self.n // STEP:05d}.png"), depth)
        self.n += 1

    def stop(self):
        pass


def best_frames(d, bg, k=10):
    """sharpest recorded frame with someone in front among the first k and among the last k -> (before, after) as
    (bgr, depth), None if nobody: the very first/last person frame is often a blur of a moving arm"""
    first, last = [], collections.deque(maxlen=k)
    for c in sorted((d / "colour").glob("*.jpg")):
        z = cv2.imread(str(d / "depth" / f"{c.stem}.png"), cv2.IMREAD_UNCHANGED)
        if z is not None and z.shape == bg.shape and person(z, bg):
            (first if len(first) < k else last).append((cv2.imread(str(c)), z))
    if not first:
        return None
    sharp = lambda f: cv2.Laplacian(cv2.cvtColor(f[0], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return max(first, key=sharp), max(last or first, key=sharp)


def save(opened, closed, before, after, cid, out=LOG):
    d = folder(opened, out)
    d.mkdir(parents=True, exist_ok=True)
    keep(d, before, after)
    stamp = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))
    meta = {"closet": cid, "opened": stamp(opened), "closed": stamp(closed), "person": before is not None}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def keep(d, before, after):
    for name, f in (("before", before), ("after", after)):
        if f:
            cv2.imwrite(str(d / f"{name}.jpg"), f[0])
            cv2.imwrite(str(d / f"{name}_depth.png"), f[1])  # 16-bit PNG keeps the mm values


MATCH = """[조각]은 옷장 앞 카메라 사진에서 잘라 낸 옷이다. [옷 N]은 이 사용자의 옷장에 등록된 옷이다.
[조각]과 같은 옷이 [옷 N] 중에 있으면 그 번호를, 없으면 0을 고른다. 색, 무늬, 종류가 모두 같아야 같은 옷이다.
[조각]이 옷이 아니면(사람 피부, 배경 등) 0을 고른다.
JSON으로만 답한다: {"desc": "조각 속 옷의 색과 종류", "match": 번호}"""


def judge(d, cid, ask=None, store=None, find=None):
    """session folder d -> garments taken out / put back.
    detect.find cuts the garments out of both shots (saved as before_seg.jpg / after_seg.jpg); Qwen names each cut-out
    against the user's closet one at a time (FashionCLIP shortlist); taken = with the user at closing but not at opening,
    returned = the reverse. Comparing the two shots was left to code: asked to, the 8B model "saw" the jacket in both.
    Writes d/meta.json and flips each garment's "out" flag. Qwen down -> meta gets "error", rerun later."""
    import web, detect
    from PIL import Image
    ask, store, find = ask or web.x.ask_qwen, store or web.UPLOADS, find or detect.find
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    mine = list(web.closet(cid, store).items())[-web.MAX_PICK:]  # ponytail: newest MAX_PICK garments only; filter by category when closets grow
    if not meta["person"] or not mine:
        return meta
    fronts = {k: Image.open(store / k / "front_cut.png").convert("RGB") for k, _ in mine}
    bg = cv2.imread(str(BG), cv2.IMREAD_UNCHANGED) if BG.exists() else None
    if bg is not None and (d / "colour").is_dir() and (best := best_frames(d, bg)):
        keep(d, *best)
    held = {}
    try:
        for name in ("before", "after"):
            im = Image.open(d / f"{name}.jpg").convert("RGB")
            z = cv2.imread(str(d / f"{name}_depth.png"), cv2.IMREAD_UNCHANGED)
            found = find(im, closer(z, bg) if bg is not None and z is not None and z.shape == bg.shape else None)
            detect.overlay(im, found).save(d / f"{name}_seg.jpg")
            held[name], meta[f"found_{name}"] = set(), []
            for g in found:
                cands = detect.shortlist([g["cut"]], list(fronts.items()))
                labels = ["[조각]"] + [f"[옷 {i}] {dict(mine)[k]['category']} · {dict(mine)[k].get('desc') or ''}" for i, (k, _) in enumerate(cands, 1)]
                try:
                    r = ask([g["cut"]] + [im_ for _, im_ in cands], prompt=MATCH, labels=labels, side=web.SIDE, max_tokens=100, grammar=False)
                except ValueError:  # one cut answered without JSON: unknown piece, not a failed session (server down still fails)
                    r = {}
                n = int(r.get("match") or 0) if str(r.get("match") or 0).isdigit() else 0
                key = cands[n - 1][0] if 1 <= n <= len(cands) else None
                held[name] |= {key} - {None}
                meta[f"found_{name}"].append({"label": g["label"], "score": round(g["score"], 2), "desc": str(r.get("desc") or "")[:100], "garment": key})
    except (*web.x.VLM_ERRORS, AttributeError) as e:
        meta["error"] = f"{type(e).__name__}: {e}"[:200]
        return _write(d, meta)
    meta.pop("error", None)
    meta.update(taken=sorted(held["after"] - held["before"]), returned=sorted(held["before"] - held["after"]))
    for keys, out in ((meta["taken"], True), (meta["returned"], False)):
        for k in keys:
            g = json.loads((store / k / "meta.json").read_text(encoding="utf-8"))
            _write(store / k, {**g, "out": out, "moved_at": meta["closed"], "moved_in": d.name})
    return _write(d, meta)


def _write(d, meta):
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def astra():
    """-> (unix time, bgr, depth mm) forever; depth is registered to the colour image.
    The depth stream (and its laser) runs only while the colour image is bright; in the dark depth is all zeros.
    Dark for IDLE_S: the colour stream stops too and is started for a quick look every POLL_S (one dark frame is
    yielded per look, so the live window, stats and --stop keep working)."""
    from openni import openni2
    openni2.initialize(OPENNI2_DIR)
    dev = openni2.Device.open_any()  # "no devices found" with the USB device listed: replug it into a rear port
    depth, color = dev.create_depth_stream(), dev.create_color_stream()
    depth.start(), depth.stop()  # ponytail: meant to switch off a laser left on by a killed run; unverified on this unit
    color.start()
    on, dark, idle = False, 0, False
    to_bgr = lambda c: cv2.cvtColor(np.frombuffer(c.get_buffer_as_uint8(), np.uint8).reshape(c.height, c.width, 3), cv2.COLOR_RGB2BGR)
    try:
        while True:
            if STOP.exists():  # python doorcam.py --stop: a clean exit, so the laser is switched off
                STOP.unlink()
                print("stop requested", flush=True)
                return
            if idle:
                time.sleep(POLL_S)
                color.start()
                for _ in range(WAKE_FRAMES):  # auto exposure settles over the first frames after a start
                    bgr = to_bgr(color.read_frame())
                if bgr.mean() <= DARK:
                    color.stop()
                    yield time.time(), bgr, np.zeros(bgr.shape[:2], np.uint16)
                    continue
                idle = False
                print("light: camera awake", flush=True)
            else:
                bgr = to_bgr(color.read_frame())
            dark = dark + 1 if bgr.mean() <= DARK else 0
            if not on and dark >= IDLE_S * 30:
                color.stop()
                idle = True
                print("dark: camera idle", flush=True)
            if not on and not dark:
                depth.start()
                try:
                    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)  # only accepted once streaming
                except openni2.OpenNIError as e:
                    print("depth not aligned to colour:", e, flush=True)
                on = True
            elif on and dark >= SHUT_FRAMES * 6:  # stays on through a door that is only briefly shut
                depth.stop()
                on = False
            if on:
                d = depth.read_frame()
                z = np.frombuffer(d.get_buffer_as_uint16(), np.uint16).reshape(d.height, d.width).copy()
            else:
                z = np.zeros(bgr.shape[:2], np.uint16)
            yield time.time(), bgr, z
    finally:
        if on:
            depth.stop()
        if not idle:
            color.stop()
        openni2.unload()
        print("camera closed, depth laser off", flush=True)


def stats(frames, bg, every=30):
    """one line per second; "centre" = median depth of the middle of the frame: on a still scene it drifts while the
    laser warms up, the only heat hint this Astra gives (it has no readable temperature)"""
    for i, f in enumerate(frames):
        if i % every == 0:
            t, bgr, z = f
            mid = z[z.shape[0] * 2 // 5:z.shape[0] * 3 // 5, z.shape[1] * 2 // 5:z.shape[1] * 3 // 5]
            centre = int(np.median(mid[mid > 0])) if (mid > 0).any() else 0
            print(f"{time.strftime('%H:%M:%S', time.localtime(t))}  valid {(z > 0).mean():.2f}  bright {bgr.mean():.0f}  "
                  f"centre {centre}mm  closer {closer(z, bg).mean():.2f}  door {'open' if door_open(z, bgr) else 'shut'}  person {person(z, bg)}", flush=True)
        yield f


def learn(frames, bg):
    bg[:] = background(frames)  # in place: sessions() holds the same array
    LOG.mkdir(exist_ok=True)
    cv2.imwrite(str(BG), bg)
    print("empty scene saved", BG, flush=True)


def show(frames, bg):
    """live window: colour (someone tinted green) | depth 0-4 m; b = learn the empty scene now, q/Esc = quit"""
    for f in frames:
        _, bgr, z = f
        heat = cv2.applyColorMap(cv2.convertScaleAbs(z, alpha=255 / 4000), cv2.COLORMAP_JET)
        heat[z == 0] = 0
        left = bgr.copy()
        m = cv2.resize(closer(z, bg).astype(np.uint8), bgr.shape[1::-1]) > 0
        left[m] = left[m] // 2 + (0, 127, 0)
        view = np.hstack([left, cv2.resize(heat, bgr.shape[1::-1])])
        text = f"door {'OPEN' if door_open(z, bgr) else 'SHUT'}  person {person(z, bg)}  closer {closer(z, bg).mean():.2f}"
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view, "b: learn empty scene   q: quit", (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("doorcam", view)
        key = cv2.waitKey(1)
        if key in (ord("q"), 27):
            return
        if key == ord("b"):
            learn(frames, bg)
        yield f


def prune(out=LOG, now=None, keep_days=KEEP_DAYS, max_bytes=MAX_RAW_GB * 1e9):
    """raw frames (colour/, depth/: ~1.8 MB/s of open door) are kept keep_days for re-analysis, then deleted;
    above max_bytes the oldest finished sessions (meta.json written) go first. The shots the judge uses
    (before/after, *_seg.jpg, meta.json: ~1 MB per opening) always stay. -> deleted session names"""
    now, raw, gone = now or time.time(), [], []
    for d in sorted(out.glob("2*-*")):  # folder names sort by time: oldest first
        subs = [s for s in (d / "colour", d / "depth") if s.is_dir()]
        try:
            t = time.mktime(time.strptime(d.name, "%Y%m%d-%H%M%S"))
        except ValueError:
            continue
        if subs:
            raw.append((t, d, subs, sum(f.stat().st_size for s in subs for f in s.iterdir())))
    total = sum(r[3] for r in raw)
    for t, d, subs, size in raw:
        if now - t > keep_days * 86400 or (total > max_bytes and (d / "meta.json").exists()):
            for s in subs:
                shutil.rmtree(s, ignore_errors=True)
            total -= size
            gone.append(d.name)
            print("pruned raw frames", d.name, f"{size / 1e6:.0f} MB", flush=True)
    return gone


def worker(out=LOG, every=5):
    """backend: analyse every recorded session with someone in it that has no result yet; sessions that failed
    (Qwen down) are retried once per worker start. Raw frames are pruned every 10 minutes."""
    tried, pruned = set(), 0
    while True:
        for m in sorted(out.glob("*/meta.json")):
            meta = json.loads(m.read_text(encoding="utf-8"))
            if meta.get("person") and "taken" not in meta and m.parent not in tried:
                tried.add(m.parent)
                r = judge(m.parent, meta["closet"])
                print("judged", m.parent.name, {k: r.get(k) for k in ("taken", "returned", "error")}, flush=True)
        if time.time() - pruned > 600:
            prune(out)
            pruned = time.time()
        time.sleep(every)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--closet")
    ap.add_argument("--show", action="store_true", help="live window")
    ap.add_argument("--worker", action="store_true", help="analyse recorded sessions (backend)")
    ap.add_argument("--stop", action="store_true", help="ask a running camera to close cleanly (laser off)")
    a = ap.parse_args()
    if a.stop:
        LOG.mkdir(exist_ok=True)
        STOP.touch()
        raise SystemExit("stop requested")
    if a.worker:
        worker()
    if not a.closet:
        ap.error("--closet is required for the camera")
    # Ctrl+C / Ctrl+Break / terminate -> exception -> astra()'s finally stops the streams. A hard kill
    # (Stop-Process -Force, TerminateProcess) can't be caught and may leave the laser on: use --stop instead.
    for sig in (signal.SIGTERM, getattr(signal, "SIGBREAK", signal.SIGTERM)):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    STOP.unlink(missing_ok=True)
    cam = astra()
    try:
        bg = cv2.imread(str(BG), cv2.IMREAD_UNCHANGED) if BG.exists() else None
        if bg is None:
            print("learning the empty scene: keep out of view for a second", flush=True)
            bg = np.zeros((480, 640), np.uint16)
            learn(cam, bg)
        frames = stats(cam, bg)
        for s in sessions(show(frames, bg) if a.show else frames, bg, Tape()):
            print("recorded", save(*s, a.closet), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()  # runs astra()'s finally even when another generator was running at the interrupt
