"""More training scenes for the 26 from camweb recordings (doorcam_web/<time>/rec/<unix time>.jpg), on this PC only.
  python more_data.py label [rec folders...]   recordings -> collect/pending (checked at /review like the collected scenes):
      filled  a held garment the 26 lost for a moment inside one move: its box drawn between the frames around it
      track   frames of a move the 26 saw anyway (every STEP-th)
      lone    a held box that came and went alone (most likely not a held garment)
      empty   now and then a frame with nothing held
  python more_data.py synth [n]                ATR garment cut-outs pasted, big and blurred, onto the empty rail frames of
      the recordings (nobody there, nothing held) -> collect/synth/images, labels; ready for training, no checking needed
No folders given: every recording not labelled yet (a folder gets rec/labelled.txt once done)."""
import io, json, random, sys
from pathlib import Path
import cv2
import numpy as np
import collect
import live

ROOT = live.ROOT
RECS, SYNTH, ATR = ROOT / "doorcam_web", collect.ROOT / "synth", ROOT / "yolo/raw/data"
# ponytail: set without a rail recording yet; check on the first one
MAX_FILL = 1.0     # s: a garment lost longer than this inside a track is not filled in
MIN_BOXES = 3      # boxes a track needs (one of them >= live.STRONG) to count as a held garment
STEP = 3           # every STEP-th seen frame of a track is kept
EMPTY_EVERY = 3.   # s between kept empty frames
WEAK = .2          # a weak box near the filled-in one is used instead (it sits better)
SAME = .3          # IoU of the boxes either side of a gap (or close and alike in size) for it to be filled


def frames(rec):
    return sorted(((float(p.stem), p) for p in rec.glob("*.jpg")), key=lambda x: x[0])


def tracks(seen):
    """seen: [(t, dets)] -> held-garment tracks, each [(index, conf, box)], linked as live.Events links them"""
    out, open_ = [], []
    for i, (t, dets) in enumerate(seen):
        open_ = [k for k in open_ if t - seen[k[-1][0]][0] <= MAX_FILL]
        for conf, b in sorted(((s, b) for c, s, b in dets if c == 0 and s >= live.CONF), reverse=True):
            near = [k for k in open_ if k[-1][0] < i and (live.iou(k[-1][2], b) > .2 or live.dist(k[-1][2], b) < .3)]
            k = max(near, key=lambda k: live.iou(k[-1][2], b), default=None)
            if k is None:
                k = []
                open_.append(k)
                out.append(k)
            k.append((i, conf, b))
    return out


def fill(seen, k):
    """{frame index: box} for the frames inside track k where the 26 lost it"""
    got = {}
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    for (i, _, a), (j, _, z) in zip(k, k[1:]):
        if live.iou(a, z) < SAME and not (live.dist(a, z) < .15 and .5 < area(a) / area(z) < 2):
            continue  # the boxes either side are not one garment moving smoothly: no guessing
        for m in range(i + 1, j):
            w = (seen[m][0] - seen[i][0]) / (seen[j][0] - seen[i][0])
            guess = tuple(np.array(a) + (np.array(z) - a) * w)
            weak = [b for c, s, b in seen[m][1] if c == 0 and WEAK <= s < live.CONF and live.iou(b, guess) > .3]
            got[m] = max(weak, key=lambda b: live.iou(b, guess)) if weak else guess
    return got


def label(rec, held=None):
    """one recording -> pending scenes; -> how many of each kind"""
    fr = frames(rec)
    seen = [(t, live.see(live.imread(p), 640, held)) for t, p in fr]
    ks = tracks(seen)
    good = [k for k in ks if len(k) >= MIN_BOXES and max(c for _, c, _ in k) >= live.STRONG]
    lone = [k for k in ks if k not in good]
    keep = {}  # frame index -> (kind, extra boxes)
    for k in good:
        for m, b in fill(seen, k).items():
            keep.setdefault(m, ("filled", []))[1].append(b)
        for n, (i, _, _) in enumerate(k):
            if n % STEP == 0:
                keep.setdefault(i, ("track", []))
    for k in lone:
        keep.setdefault(k[0][0], ("lone", []))
    busy = {i for k in ks for i in range(k[0][0] - 3, k[-1][0] + 4)}
    last = -1e9
    for i, (t, dets) in enumerate(seen):
        if i not in busy and t - last >= EMPTY_EVERY and not any(c == 0 and s >= WEAK for c, s, _ in dets):
            keep.setdefault(i, ("empty", []))
            last = t
    s = collect.PENDING / rec.parent.name[:15]
    s.mkdir(parents=True, exist_ok=True)
    n = len(list(s.glob("*.json")))
    order = sorted(keep, key=lambda i: (("filled", "lone", "track", "empty").index(keep[i][0]), i))  # /review shows filled first
    for i in order:
        kind, extra = keep[i]
        n += 1
        boxes = [[c, round(p, 3), *(round(v, 4) for v in b)] for c, p, b in seen[i][1] if p >= collect.KEEP[c]]
        boxes += [[0, 0.0, *(round(float(v), 4) for v in b)] for b in extra]  # conf 0 = filled in, /review shows it as such
        (s / f"{n:05d}.jpg").write_bytes(fr[i][1].read_bytes())
        (s / f"{n:05d}.json").write_text(json.dumps({"kind": kind, "time": fr[i][0], "model": "자동 라벨", "boxes": boxes}, ensure_ascii=False))
    (rec / "labelled.txt").write_text("done")
    return {k: sum(v[0] == k for v in keep.values()) for k in ("filled", "track", "lone", "empty")}


def cutouts(n, seed=0):
    """ATR garment cut-outs (image, mask)"""
    import pyarrow.parquet as pq
    from PIL import Image
    rng, out = random.Random(seed), []
    for f in sorted(ATR.glob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            for img, msk in zip(batch.column("image").to_pylist(), batch.column("mask").to_pylist()):
                if rng.random() > .3:
                    continue
                m = np.array(Image.open(io.BytesIO(msk["bytes"])))
                for c in (4, 7, 6, 5):  # upper clothes, coat, pants, skirt
                    k = (m == c).astype(np.uint8)
                    if k.mean() < .04:
                        continue
                    a = np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))[..., ::-1]
                    ys, xs = np.where(k)
                    s = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                    out.append((a[s].copy(), k[s] * 255))
                    break
                if len(out) >= n:
                    return out
    return out


def paste(bg, cut, rng):
    """cut pasted big, turned, maybe cut off by the frame and motion-blurred -> (image, mask) or None"""
    a, k = cut
    H, W = bg.shape[:2]
    s = rng.uniform(.4, 1.2) * H / a.shape[0]
    a = cv2.resize(a, None, fx=s * rng.uniform(.8, 1.3), fy=s)
    k = cv2.resize(k, a.shape[1::-1])
    d = int(np.hypot(*a.shape[:2]))
    M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), rng.uniform(-25, 25), 1)
    M[:, 2] += (d / 2 - a.shape[1] / 2, d / 2 - a.shape[0] / 2)
    a, k = cv2.warpAffine(a, M, (d, d)), cv2.warpAffine(k, M, (d, d))
    x, y = int(rng.uniform(0, 1) * W - d / 2), int(rng.uniform(.1, 1) * H - d / 2)
    fa, fk = np.zeros_like(bg), np.zeros((H, W), np.uint8)
    X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + d, W), min(y + d, H)
    if X1 - X0 < 40 or Y1 - Y0 < 40:
        return None
    fa[Y0:Y1, X0:X1], fk[Y0:Y1, X0:X1] = a[Y0 - y:Y1 - y, X0 - x:X1 - x], k[Y0 - y:Y1 - y, X0 - x:X1 - x]
    fa = np.clip(fa.astype(np.float32) * rng.uniform(.5, 1.4), 0, 255).astype(np.uint8)
    if L := rng.choice([0, 0, 9, 19, 31]):
        ker = np.zeros((L, L), np.float32)
        ker[L // 2] = 1 / L
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((L / 2, L / 2), rng.uniform(0, 180), 1), (L, L))
        ker /= ker.sum() + 1e-6
        fa, fk = cv2.filter2D(fa, -1, ker), cv2.filter2D(fk, -1, ker)
    if (fk > 127).mean() < .03:
        return None
    al = (fk / 255.)[..., None]
    return (bg * (1 - al) + fa * al).astype(np.uint8), fk > 127


def backgrounds(recs, every=2.):
    """empty rail frames: nobody (pose) and nothing held, one per `every` s -> [(bgr, the 26's hanger / arm boxes)]"""
    out = []
    for rec in recs:
        last = -1e9
        for t, p in frames(rec):
            if t - last < every:
                continue
            bgr = live.imread(p)
            dets = live.see(bgr, 640)
            if any(c == 0 and s >= WEAK for c, s, _ in dets) or len(live.yolo(live.POSE)(bgr, imgsz=640, conf=.25, verbose=False)[0].boxes):
                continue
            out.append((bgr, [(c, b) for c, s, b in dets if c == 2 and s >= collect.KEEP[2]]))
            last = t
    return out


def synth(n, recs, seed=0):
    bgs = backgrounds(recs)
    if not bgs:
        raise SystemExit("사람도 든 옷도 없는 녹화 장면이 없습니다. 옷걸이만 찍힌 몇 초를 녹화해 주세요.")
    cuts, rng = cutouts(min(n, 1200), seed), random.Random(seed)
    (SYNTH / "images").mkdir(parents=True, exist_ok=True)
    (SYNTH / "labels").mkdir(exist_ok=True)
    i = len(list((SYNTH / "images").glob("*.jpg")))
    made = 0
    while made < n:
        bg, hangers = rng.choice(bgs)
        r = paste(bg, rng.choice(cuts), rng)
        if r is None:
            continue
        im, m = r
        H, W = m.shape
        ys, xs = np.where(m)
        rows = [(0, xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H)]
        for c, (x0, y0, x1, y1) in hangers:  # a hanger mostly hidden by the pasted garment is not labelled
            if m[int(y0 * H):int(np.ceil(y1 * H)), int(x0 * W):int(np.ceil(x1 * W))].mean() < .5:
                rows.append((c, x0, y0, x1, y1))
        cv2.imwrite(str(SYNTH / f"images/r{i:05d}.jpg"), im)
        (SYNTH / f"labels/r{i:05d}.txt").write_text("\n".join(f"{c} {(a + z) / 2:.5f} {(b + w) / 2:.5f} {z - a:.5f} {w - b:.5f}" for c, a, b, z, w in rows))
        i, made = i + 1, made + 1
    (SYNTH / "d.yaml").write_text(f"path: {SYNTH.as_posix()}\ntrain: images\nval: images\nnames:\n"
                                  + "".join(f"  {k}: {v}\n" for k, v in enumerate(collect.NAMES)))
    return made, len(bgs)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("label", "synth"):
        raise SystemExit(__doc__)
    all_recs = sorted(p for p in RECS.glob("*/rec") if any(p.glob("*.jpg")))
    if sys.argv[1] == "label":
        recs = [Path(a) for a in sys.argv[2:]] or [r for r in all_recs if not (r / "labelled.txt").exists()]
        for r in recs:
            print(r.parent.name, label(r), flush=True)
    else:
        made, nb = synth(int(sys.argv[2]) if len(sys.argv) > 2 else 2000, all_recs)
        print(f"합성 {made}장 (배경 {nb}장) -> {SYNTH}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
