"""Closet door, live: a garment is carried in or out -> which way it went, registered / unregistered / unsure, when (track.py).
Any camera whose picture has the closet on its left or right edge and the user on the other, e.g. a laptop's own camera.
  python live.py --add 이름 앞.jpg 뒤.jpg 옆.jpg   register a garment (3 photos)
  python live.py --side right                      run with the laptop camera; the closet is on the right of the picture (q: quit)
  python live.py --state                           what is in the closet, what is out, and since when
  python live.py --resolve U2 이름|미등록            settle an unsure garment"""
import argparse, json, queue, shutil, sys, threading, time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import db
import match
import vl
from track import NEW, REGISTERED, UNSURE, Track

ROOT = Path(__file__).resolve().parent
OUT, MINE, UPLOADS, HELD, POSE = ROOT / "doorcam", ROOT / "registered", ROOT / "uploads", ROOT / "models" / "held.pt", ROOT / "match" / "pose.pt"
NAMES = ("옷", "팔", "옷걸이")  # the 26's classes
EVENTS = ROOT / "events.jsonl"  # every call the camera makes, all sessions, for the other parts of Fitmate (camweb /api/events)
_events = threading.Lock()
if not HELD.exists():  # models/held.pt: this PC's own 26 (fine-tuned on its closet scenes, never on GitHub); else the public one
    HELD = ROOT / "match" / "held.pt"
VIEWS = ("front", "back", "side")
SEE_EVERY = .1     # s between looked-at frames
# ponytail: set on one 45 s video; check on the mounted camera
CONF = .35         # a held garment in one frame
STRONG = .7        # ...and at least one frame as sure as this
MIN_SEEN = 5       # frames
GAP = .7           # s without it: it has gone
MOVED = .12        # how far it must have gone (share of the frame) to count as in or out
LOWER = .6         # weight of up / down against towards / away from the closet: garments come in from below and leave downwards


def iou(a, b):
    i = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def dist(a, b):
    return np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)


def best(k, n=5):
    """the clearest, fullest view in each fifth of the move"""
    good = lambda x: x[1] * min(1, (x[2][2] - x[2][0]) * (x[2][3] - x[2][1]) / .15) ** .5
    return [max(p, key=good)[3] for i in np.array_split(np.arange(len(k)), min(n, len(k))) if (p := [k[j] for j in i if k[j][3].size])]


class Events:
    """boxes of held garments, frame by frame -> finished moves (action, start, end, [crops])"""
    def __init__(self, side):
        self.sign, self.tracks = 1 if side == "right" else -1, []

    def step(self, t, boxes, bgr):
        """boxes: [(conf, (x0, y0, x1, y1) as a share of the frame)]"""
        done = [k for k in self.tracks if t - k[-1][0] > GAP]
        self.tracks = [k for k in self.tracks if t - k[-1][0] <= GAP]
        for conf, b in sorted((q for q in boxes if q[0] >= CONF), reverse=True):  # a weak box (worn clothes) on a track bends its path
            near = [k for k in self.tracks if k[-1][0] < t and (iou(k[-1][2], b) > .2 or dist(k[-1][2], b) < .3)]
            k = max(near, key=lambda k: iou(k[-1][2], b), default=None)
            if k is None:
                k = []
                self.tracks.append(k)
            h, w = bgr.shape[:2]
            x0, y0, x1, y1 = int(b[0] * w), int(b[1] * h), int(np.ceil(b[2] * w)), int(np.ceil(b[3] * h))
            k.append((t, conf, b, bgr[y0:y1, x0:x1].copy()))
            if len(k) > 300:  # ponytail: a garment in view for 30 s keeps only its first and last frames
                del k[10:110]
        return [m for k in done if (m := self.move(k))]

    def flush(self):
        done, self.tracks = self.tracks, []
        return [m for k in done if (m := self.move(k))]

    def move(self, k):
        if len(k) < MIN_SEEN or max(c for _, c, _, _ in k) < STRONG:
            return None
        n = max(2, len(k) // 10)  # where it was first and last, the surer boxes counting more
        a, z = (np.average([b for _, _, b, _ in q], 0, [c * c for _, c, _, _ in q]) for q in (k[:n], k[-n:]))
        towards = self.sign * ((z[0] + z[2]) - (a[0] + a[2])) / 2 - LOWER * (z[1] - a[1])
        if abs(towards) < MOVED:  # held up and put back, or nothing that moved
            return None
        return ("in" if towards > 0 else "out"), k[0][0], k[-1][0], best(k)


def imread(p):  # cv2.imread fails silently on Korean paths
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def photos(d):
    """registration photos of one garment, front / back / side"""
    return [p for v in VIEWS for p in sorted(d.glob(f"{v}.*"))[:1]]


def stamp(t):
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def event(out, held, action, start, end, crop, e=None, **more):
    """one call as the other parts read it (README: 판정 로그). e: the move from Track.move (None: not a garment)"""
    c = Path(out) / crop
    v = {"time": stamp(end), "start": stamp(start), "session": Path(out).name, "model": Path(held).stem, "action": action,
         "result": "옷 아님", "garment": None, "db_id": None, "p": None, "candidates": [], "twin": None, "qwen": None,
         "crop": c.relative_to(ROOT).as_posix() if c.is_relative_to(ROOT) else str(c), "compare": [], **more}
    if e:
        ids = db.ids()
        v.update(result=e["kind"], garment=e["item"], db_id=ids.get(e["item"]), p=e["p"], candidates=e["candidates"], twin=e["twin"],
                 compare=[{**x, "db_ids": [ids.get(k) for k in x["alike"]]} for x in e.get("compare", [])],
                 qwen={"answer": e["vl"], "without": e["without"], "why": e.get("vl_why")} if "vl" in e else None)
    with _events, EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(v, ensure_ascii=False) + "\n")
    return v


def garments(cid=None):
    """{key: folder}: registered here with --add, the user's garments in the team DB (db.py, last pull) and the garments of one
    closet of the Fitmate server"""
    g = {d.name: d for d in sorted(MINE.glob("*")) if d.is_dir() and photos(d)}
    g.update({k: d for k, d in db.garments(g).items() if photos(d)})
    for m in sorted(UPLOADS.glob("*/*/meta.json")) if cid else []:
        meta = json.loads(m.read_text(encoding="utf-8"))
        if meta.get("closet") == cid and meta.get("in_closet"):
            g[f"{m.parent.parent.name}/{m.parent.name}"] = m.parent
    return g


def registered(cid=None):
    """{key: embeddings of its photos}; kept in doorcam/emb until a photo changes"""
    out = {}
    for key, d in garments(cid).items():
        ps, f = photos(d), OUT / "emb" / (key.replace("/", "__") + ".npz")
        if not f.exists() or f.stat().st_mtime < max(p.stat().st_mtime for p in ps):
            print(f"등록 사진 읽는 중: {key} ({len(ps)}장)", flush=True)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, *match.embed([match.cut_out(imread(p)) for p in ps]))
        with np.load(f) as z:
            out[key] = [z[k] for k in z.files]
    return out


def add(name, files):
    ims = [imread(f) for f in files if Path(f).exists()]
    if len(ims) != 3 or any(i is None for i in ims):
        raise SystemExit("사진 3장(앞, 뒤, 옆)을 읽을 수 없습니다: " + " ".join(files))
    if not name.strip() or any(c in name for c in '\\/:*?"<>|.') or (name[0] == "U" and name[1:].isdigit()) or name == NEW:
        raise SystemExit("쓸 수 없는 이름입니다: " + name)
    d = MINE / name
    if d.exists():
        raise SystemExit("이미 등록된 이름입니다: " + name)
    d.mkdir(parents=True)
    for v, f in zip(VIEWS, files):
        shutil.copy(f, d / (v + Path(f).suffix.lower()))
    for p in photos(d):  # newer than any embedding kept under the same name
        p.touch()
    registered()
    print("등록 완료:", name)


@lru_cache(None)
def yolo(f=None):
    from ultralytics import YOLO
    return YOLO(str(f or HELD))


def is_person(b, faces):
    """a box around a face is the user, not a garment; a wide one may be a garment held up beside the face.
    faces: per person, the (x, y) of the nose, eyes and ears that were seen"""
    return b[2] - b[0] < .6 and any(sum(b[0] <= x <= b[2] and b[1] <= y <= b[3] for x, y in f) >= 2 for f in faces)


SEEING = threading.Lock()  # ponytail: one frame at a time for every viewer of camweb.py; a GPU or a second process if it lags


def see(bgr, size, held=None):
    """-> [(class, conf, box as a share of the frame)]; classes 0 held garment, 1 arm, 2 hanger"""
    with SEEING:
        return _see(bgr, size, held)


def _see(bgr, size, held):
    h, w = bgr.shape[:2]
    b = yolo(held)(bgr, imgsz=size, conf=.2, verbose=False)[0].boxes
    out = [(int(c), float(s), (x[0] / w, x[1] / h, x[2] / w, x[3] / h)) for c, s, x in zip(b.cls, b.conf, b.xyxy.tolist())]
    if any(c == 0 for c, _, _ in out):
        r = yolo(POSE)(bgr, imgsz=640, conf=.25, verbose=False)[0]
        faces = [[(x / w, y / h) for (x, y), k in zip(xy[:5], cf[:5]) if k > .5] for xy, cf in zip(r.keypoints.xy.tolist(), r.keypoints.conf.tolist())] if len(r.boxes) else []
        out = [d for d in out if d[0] != 0 or not is_person(d[2], faces)]
    return out


class Camera(threading.Thread):
    """the newest frame only, so that a slow detector never lags behind"""
    def __init__(self, index, size=(1280, 720)):
        super().__init__(daemon=True)
        for i in [index] + [j for j in range(4) if j != index]:
            self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            if self.cap.isOpened() and self.cap.read()[0]:
                print(f"카메라 {i}번: {int(self.cap.get(3))}x{int(self.cap.get(4))}", flush=True)
                break
            self.cap.release()
        else:
            raise SystemExit("카메라를 열 수 없습니다. 다른 프로그램(줌 등)이 쓰고 있지 않은지, Windows 설정 > 개인 정보 > 카메라가 켜져 있는지 확인하세요.")
        self.now, self.stop = None, False
        self.start()

    def run(self):
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                self.now = ()  # unplugged
                break
            self.now = (time.time(), f)
        self.cap.release()

    def frames(self):
        last = 0
        try:
            while True:
                now = self.now
                if now == ():
                    raise RuntimeError("카메라 연결이 끊겼습니다.")
                if now is None or now[0] - last < SEE_EVERY:
                    time.sleep(.005)
                    continue
                last = now[0]
                yield now
        finally:
            self.stop = True
            self.join(2)


def video(path):
    """a recorded file as if it had just been live"""
    cap, i, last, t0 = cv2.VideoCapture(str(path)), 0, -9, time.time()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    while True:
        ok, f = cap.read()
        if not ok:
            return
        if i / fps - last >= SEE_EVERY - 1e-6:
            last = i / fps
            yield t0 + last, f
        i += 1


@lru_cache(None)
def font(px):
    from PIL import ImageFont
    for f in ("malgun.ttf", "AppleGothic.ttf", "NanumGothic.ttf"):
        try:
            return ImageFont.truetype(f, px)
        except OSError:
            pass
    return ImageFont.load_default(px)


def draw(bgr, dets, lines):
    from PIL import Image, ImageDraw
    im = cv2.resize(bgr, (960, round(960 * bgr.shape[0] / bgr.shape[1])))
    h, w = im.shape[:2]
    for c, s, b in dets:
        if s >= CONF:
            cv2.rectangle(im, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), ((0, 200, 0), (255, 120, 0), (0, 0, 255))[c], 2)
    im = Image.fromarray(im[..., ::-1])
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        d.text((10, 8 + 26 * i), l, font=font(20), fill="white", stroke_width=2, stroke_fill="black")
    return np.array(im)[..., ::-1]


def who(e):
    return {REGISTERED: e["item"], NEW: e["item"], UNSURE: f"{e['item']} ({', '.join(e['candidates'])} 일 수 있음)"}[e["kind"]]


def line(e):
    """one move for the screen and the log; asked to Qwen: what the matcher alone said beside what Qwen made of it"""
    s = f"{e['to'][11:19]} {'입고' if e['action'] == 'in' else '출고'} · {e['kind']} · {who(e)} · {e['p']:.2f}{' · ' + e['twin'] + '와 같은 옷 한 벌 더' if e['twin'] else ''}"
    return s + (f" · [Qwen 없이: 확인필요 {e['without']} | Qwen: {e['vl']}]" if "vl" in e else "")


def run(frames, side, cid=None, size=640, window=True, out=OUT, wait=False, closet=None, held=None, on_frame=None, on_move=None):
    """frames: (unix time, bgr). wait: finish each comparison before the next frame (recorded files).
    held: which 26. on_frame(dets, lines, judging): after every frame; on_move(action, start, end): when a move is found (camweb.py)"""
    held = Path(held or HELD)  # read HELD now, not at import: tests swap live.HELD
    for f in (held, POSE, *(match.W / f"match_{m}.pt" for m in match.MODELS), match.W / "decider.npz"):
        if not f.exists() or f.stat().st_size < 10_000:  # a Git LFS pointer, not the file
            raise SystemExit(f"모델 파일이 없습니다: {f.name}  ->  git lfs install 후 git lfs pull 을 실행하세요.")
    print("모델 여는 중...", flush=True)
    for f in (held, POSE):
        yolo(f)(np.zeros((720, 1280, 3), np.uint8), verbose=False)  # load now, not at the first garment
    match.nets()
    if db.on() and closet is None:
        try:
            print(f"DB에서 옷 {db.pull([d.name for d in MINE.glob('*')])}벌 받음", flush=True)
        except Exception as x:  # no network: the last pull is used
            print("DB 연결 실패, 마지막으로 받은 목록을 씀:", type(x).__name__, x, flush=True)
    mine = registered(cid) if closet is None else closet
    t = Track(mine, out, match.decide, vl.MARGIN if vl.on() else match.MARGIN, match.twins(mine))
    folders = garments(cid) if closet is None else {}  # registration photos, for the VL
    print(f"등록된 옷 {len(t.registered)}벌. 시작합니다.", flush=True)
    ev, todo, lines = Events(side), queue.Queue(), [f"등록 {len(t.registered)}벌 · 옷장은 {'오른쪽' if side == 'right' else '왼쪽'} · q: 종료"]

    def judge():
        while (job := todo.get()) is not None:
            action, start, end, crops = job
            try:
                name = f"track/{time.strftime('%Y%m%d-%H%M%S', time.localtime(end))}-{int(end % 1 * 1000):03d}.jpg"
                (Path(out) / "track").mkdir(parents=True, exist_ok=True)
                (Path(out) / name).write_bytes(cv2.imencode(".jpg", np.hstack([match.square(c) for c in crops]))[1].tobytes())
                frames = match.embed(crops)
                ok = match.is_garment(frames) >= .5
                if ok.mean() < .5:
                    print(f"옷이 아니어서 버림: {name} ({ok.sum()}/{len(ok)}장)", flush=True)
                    event(out, held, action, start, end, name, garment_frames=f"{ok.sum()}/{len(ok)}")
                    continue
                e = t.move(action, [x[ok] for x in frames], start, end, crop=name)
                if e is None:  # the same garment going the same way again: still that move
                    continue
                cands = {k: [imread(p) for p in photos(folders[k])] for k in e["candidates"] if k in folders}
                if e["kind"] == UNSURE and cands and vl.on():
                    try:
                        why = []
                        ans = vl.ask(cv2.imread(str(Path(out) / name)), cands, why=why)
                        e = {**e, "vl_why": why[0] if why else None}
                        said = ans or "모름"
                    except Exception as x:  # the user is asked, as without the VL
                        print("VL 실패:", type(x).__name__, x, flush=True)
                        ans, said = None, "실패"
                    if ans not in (None, NEW, e["candidates"][0]):  # it named another garment than the most alike: the user decides
                        ans, said = None, f"{ans}? (1순위와 달라 사용자에게)"
                    without = who(e)
                    if ans:
                        t.resolve(e["item"], ans, by="qwen")
                        e = {**e, "item": e["item"] if ans == NEW else ans, "kind": NEW if ans == NEW else REGISTERED, "candidates": []}
                    e = {**e, "vl": said, "without": without}
                print(line(e), flush=True)
                event(out, held, action, start, end, name, e)
                if db.on():
                    try:
                        db.push(e)
                    except Exception as x:  # the move is in track.jsonl anyway
                        print("DB 기록 실패:", type(x).__name__, x, flush=True)
                lines[1:] = [line(e)] + lines[1:6]
            except Exception as x:  # one bad move must not end the watch
                print("판정 실패:", type(x).__name__, x, flush=True)
            finally:
                todo.task_done()

    w = threading.Thread(target=judge, daemon=True)
    w.start()
    Path(out).mkdir(parents=True, exist_ok=True)
    boxes = (Path(out) / "dets.jsonl").open("a", encoding="utf-8")  # every frame's boxes, for the other parts (camweb /api/detections)
    try:
        for now, bgr in frames:
            dets = see(bgr, size, held)
            boxes.write(json.dumps({"time": stamp(now), "frame": round(now, 3), "boxes": [[NAMES[c], round(q, 3), *(round(x, 4) for x in b)] for c, q, b in dets]}, ensure_ascii=False) + "\n")
            boxes.flush()
            for m in ev.step(now, [(s, b) for c, s, b in dets if c == 0], bgr):
                todo.put(m)
                if on_move:
                    on_move(*m[:3])
            if wait:
                todo.join()
            if on_frame:
                on_frame(dets, list(lines), todo.unfinished_tasks > 0)
            if window:
                cv2.imshow("closet", draw(bgr, dets, lines + (["판정 중..."] if todo.unfinished_tasks else [])))
                if cv2.waitKey(1) in (ord("q"), 27) or cv2.getWindowProperty("closet", cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        boxes.close()
        frames.close()
        for m in ev.flush():
            todo.put(m)
        todo.put(None)
        w.join()
        if window:
            cv2.destroyAllWindows()
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=("left", "right"), default="right", help="which side of the picture the closet is on")
    ap.add_argument("--cam", type=int, default=0, help="0 = the laptop's own camera")
    ap.add_argument("--video", help="a recorded file instead of the camera")
    ap.add_argument("--imgsz", type=int, default=640, help="what the detector looks at; 640 scored best")
    ap.add_argument("--closet", help="also the garments of this closet of the Fitmate server (uploads/)")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--add", nargs=4, metavar=("이름", "앞", "뒤", "옆"))
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--resolve", nargs=2, metavar=("U번호", "이름"))
    a = ap.parse_args()
    if a.add:
        return add(a.add[0], a.add[1:])
    if a.state or a.resolve:
        t = Track({k: None for k in garments(a.closet)}, OUT, None, 0)
        if a.resolve:
            try:
                t.resolve(*a.resolve)
            except AssertionError as x:
                raise SystemExit(f"없는 이름입니다: {x}")
        for k, s in t.items().items():
            print(f"{'안' if s['inside'] else '밖'}  {s['kind']}  {k}  {s['since'] or ''}  {('후보: ' + ', '.join(s['maybe'])) if s['maybe'] else ''}")
        return
    run(video(a.video) if a.video else Camera(a.cam).frames(), a.side, a.closet, a.imgsz, not a.no_window, wait=bool(a.video))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
