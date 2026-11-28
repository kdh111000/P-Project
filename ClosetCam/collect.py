"""Training frames from the camweb.py page, kept on this PC only and used only after a person has looked at them.
A session that ticked "학습용 장면 저장" keeps, while the camera holds still:
  - frames around every in/out move (the 26's own boxes as the labels to check)
  - now and then a frame with no garment at all (so that the next 26 learns what is not one)
collect/pending/<session>/<n>.jpg + .json  ->  /review: 맞음 / 옷 아님 / 버림  ->  collect/approved/images, labels (YOLO format)"""
import json, re, time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent / "collect"
PENDING, APPROVED = ROOT / "pending", ROOT / "approved"
NAMES = ("held", "arm", "hanger")
# ponytail: set on one fixed 45 s video and the same video shaken by hand-like jitter; check on more cameras
STILL = 1.5        # px (960 wide): the background's median move between frames, at most, over a move
EDGE = .5          # s of frames kept before and after a move
PER_MOVE = 8       # frames kept per move
EMPTY_EVERY = 5    # s between kept frames with no garment...
EMPTY_CLEAR = 2    # ...and no garment for this long before it either
MAX_EMPTY = 100    # per session
KEEP = (.35, .5, .5)  # least confidence for a box to be written as a label, per class
ID = re.compile(r"\d{8}-\d{6}/\d{5}")


def _grey(bgr):
    return cv2.cvtColor(cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def shift(a, b, boxes):
    """how far the background moved from frame a to b, px at 960 wide. The garment, the arm (boxes, a share of the
    frame) are left out; the rest is mostly the room, which moves only when the camera does"""
    f = cv2.calcOpticalFlowFarneback(_grey(a), _grey(b), None, .5, 3, 15, 3, 5, 1.2, 0)
    m = np.ones((180, 320), bool)
    for x0, y0, x1, y1 in boxes:
        m[max(0, int((y0 - .05) * 180)):int((y1 + .05) * 180), max(0, int((x0 - .05) * 320)):int((x1 + .05) * 320)] = False
    if m.mean() < .15:  # boxes everywhere: all of it, then
        m[:] = True
    return float(np.hypot(np.median(f[..., 0][m]), np.median(f[..., 1][m]))) * 960 / 320


class Session:
    def __init__(self, model):
        self.dir, self.model = PENDING / time.strftime("%Y%m%d-%H%M%S"), model
        self.buf, self.prev, self.n, self.empty, self.last_empty, self.last_held = [], None, 0, 0, 0., 0.

    def frame(self, t, jpg, bgr, dets):
        """every frame the page sent: (time, its JPEG, decoded, the 26's boxes [(class, conf, box)])"""
        boxes = [b for _, p, b in dets if p >= .25]
        s = shift(self.prev[0], bgr, self.prev[1] + boxes) if self.prev else 0.
        self.prev = bgr, boxes
        self.buf = [f for f in self.buf if t - f[0] < 40] + [(t, jpg, dets, s)]  # a move lasts 30 s at most (live.Events)
        if any(c == 0 and p >= .2 for c, p, _ in dets):
            self.last_held = t
        elif (t - self.last_held >= EMPTY_CLEAR and t - self.last_empty >= EMPTY_EVERY and self.empty < MAX_EMPTY
              and len(self.buf) > 5 and np.median([f[3] for f in self.buf if t - f[0] < 3]) < STILL):
            self.last_empty, self.empty = t, self.empty + 1
            self._save([self.buf[-1]], "empty")

    def move(self, action, start, end):
        """a move live.py found: keep its frames if the camera held still. -> kept or not"""
        fr = [f for f in self.buf if start - EDGE <= f[0] <= end + EDGE]
        if len(fr) < 3 or np.median([f[3] for f in fr[1:]]) >= STILL:
            return False
        self._save([fr[i] for i in np.unique(np.linspace(0, len(fr) - 1, PER_MOVE).round().astype(int))], action)
        return True

    def _save(self, frames, kind):
        self.dir.mkdir(parents=True, exist_ok=True)
        for t, jpg, dets, _ in frames:
            self.n += 1
            p = self.dir / f"{self.n:05d}"
            p.with_suffix(".jpg").write_bytes(jpg)
            boxes = [[c, round(s, 3), *(round(v, 4) for v in b)] for c, s, b in dets if s >= KEEP[c]]
            p.with_suffix(".json").write_text(json.dumps({"kind": kind, "time": t, "model": self.model, "boxes": boxes}, ensure_ascii=False))


def pending():
    return sorted(f"{p.parent.name}/{p.stem}" for p in PENDING.glob("*/*.json"))


def item(i):
    """-> (jpg path, meta); i comes from the page, so only the shape pending() makes"""
    if not ID.fullmatch(i or "") or not (PENDING / f"{i}.json").exists():
        raise KeyError(i)
    return PENDING / f"{i}.jpg", json.loads((PENDING / f"{i}.json").read_text(encoding="utf-8"))


def _box(b):
    """one box from the page: [class, x0, y0, x1, y1] as shares of the frame"""
    c, x0, y0, x1, y1 = int(b[0]), *(min(max(float(v), 0.), 1.) for v in b[1:5])
    if len(b) != 5 or c not in (0, 1, 2) or x1 - x0 < .005 or y1 - y0 < .005:
        raise ValueError(b)
    return c, x0, y0, x1, y1


def judge(i, verdict, boxes=None):
    """맞음 (ok): the boxes as they are; 옷 아님 (nogarment): the same frame without its held-garment boxes; 버림 (drop): gone.
    boxes: the person's own [[class, x0, y0, x1, y1]] (some of the 26's struck out, missed ones drawn) instead of the 26's"""
    jpg, meta = item(i)
    boxes = [b[:1] + b[2:] for b in meta["boxes"]] if boxes is None else [_box(b) for b in boxes[:50]]
    if verdict in ("ok", "nogarment"):
        (APPROVED / "images").mkdir(parents=True, exist_ok=True)
        (APPROVED / "labels").mkdir(exist_ok=True)
        (APPROVED / "d.yaml").write_text(f"path: {APPROVED.as_posix()}\ntrain: images\nval: images\nnames:\n"
                                         + "".join(f"  {k}: {n}\n" for k, n in enumerate(NAMES)))
        name = i.replace("/", "_")
        rows = [f"{c} {(x0 + x1) / 2:.5f} {(y0 + y1) / 2:.5f} {x1 - x0:.5f} {y1 - y0:.5f}"
                for c, x0, y0, x1, y1 in boxes if verdict == "ok" or c != 0]
        (APPROVED / "labels" / f"{name}.txt").write_text("\n".join(rows))
        jpg.replace(APPROVED / "images" / f"{name}.jpg")
    elif verdict == "drop":
        jpg.unlink()
    else:
        raise ValueError(verdict)
    (PENDING / f"{i}.json").unlink()
    if not any(jpg.parent.iterdir()):
        jpg.parent.rmdir()


def approved():
    return len(list((APPROVED / "images").glob("*.jpg"))) if (APPROVED / "images").exists() else 0


if __name__ == "__main__":  # self-check on made-up frames, in a scratch folder
    import tempfile
    ROOT = Path(tempfile.mkdtemp())
    PENDING, APPROVED = ROOT / "pending", ROOT / "approved"
    rng = np.random.default_rng(0)
    room = cv2.GaussianBlur(rng.integers(0, 255, (540, 960, 3), np.uint8), (0, 0), 3)
    jpg = cv2.imencode(".jpg", room)[1].tobytes()
    assert shift(room, room, []) < .5 and shift(room, np.roll(room, 12, 1), []) > 8  # held still / camera moved 12 px
    s = Session("test")
    held = [(0, .9, (.4, .3, .6, .8)), (1, .7, (.55, .4, .7, .6)), (0, .3, (.1, .1, .2, .2))]
    for k in range(40):  # 4 s of a garment, then 8 s of nothing, camera still
        s.frame(k * .1, jpg, room, held if k < 40 else [])
    for k in range(40, 120):
        s.frame(k * .1, jpg, room, [])
    assert s.move("in", 0, 3.9) and s.empty >= 1
    shaken = Session("test")
    for k in range(40):
        shaken.frame(k * .1, jpg, np.roll(room, 15 * (k % 2), 1), held)
    assert not shaken.move("in", 0, 3.9)
    ids = pending()
    a = next(i for i in ids if item(i)[1]["kind"] == "in")
    assert [b[0] for b in item(a)[1]["boxes"]] == [0, 1]  # the weak 0.3 box is not a label
    judge(a, "ok")
    b = next(i for i in pending() if item(i)[1]["kind"] == "in")
    judge(b, "nogarment")
    c = next(i for i in pending() if item(i)[1]["kind"] == "in")
    judge(c, "ok", [[0, .1, .2, .3, .9], [1, -.1, .5, .2, 1.4]])  # drawn by hand; the second runs off the frame
    assert (APPROVED / "labels" / f"{c.replace('/', '_')}.txt").read_text() == "0 0.20000 0.55000 0.20000 0.70000\n1 0.10000 0.75000 0.20000 0.50000"
    for bad in ([[3, 0, 0, .5, .5]], [[0, .5, .5, .5, .6]], [[0, .1]], "x"):
        try:
            judge(pending()[0], "ok", bad)
            raise AssertionError(bad)
        except (ValueError, TypeError, IndexError):
            pass
    judge(pending()[0], "drop")
    lab = sorted((APPROVED / "labels").glob("*.txt"))
    assert sorted(p.read_text().split("\n")[0][0] for p in lab) == ["0", "0", "1"] and approved() == 3
    try:
        item("../../web")
        raise AssertionError
    except KeyError:
        pass
    print("ok")
