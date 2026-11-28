"""Fine-tune the 26 on the closet scenes checked by hand (collect/approved), all on this PC.
  python finetune.py test          train on the first session, score on the others (scenes it has not seen)
  python finetune.py final <name>  train on every checked scene -> models/<name>.pt
Real scenes x5 + 200 synthetic ones from the 26's own data (collect/synth_base: so that what it knew is not forgotten)
+ the rail synthetic scenes of more_data.py synth (collect/synth), if any. Base: match/held.pt (6차, public). CPU. To use the result: copy it to models/held.pt."""
import shutil, sys
from pathlib import Path
from ultralytics import YOLO
import live

C = Path(__file__).resolve().parent
A, BASE, W = C / "collect/approved", C / "match/held.pt", C / "collect/ft"
SYN = {"b": (C / "collect/synth_base", C / "collect/synth_base"), "r": (C / "collect/synth/images", C / "collect/synth/labels")}


def boxes(p):
    return [(int(r.split()[0]), *map(float, r.split()[1:])) for r in (A / "labels" / f"{p.stem}.txt").read_text().split("\n") if r.strip()]


def score(m, ps):
    """found: hand-labelled held garments matched (IoU .5); fired: scenes without one where the 26 still saw one"""
    found, fired = [], []
    for p in ps:
        r = m(str(p), imgsz=640, conf=.25, verbose=False)[0].boxes
        got = [x for c, x in zip(r.cls.tolist(), r.xyxyn.tolist()) if int(c) == 0]
        gt = [(x - w / 2, y - h / 2, x + w / 2, y + h / 2) for c, x, y, w, h in boxes(p) if c == 0]
        found += [any(live.iou(g, x) >= .5 for x in got) for g in gt]
        if not gt:
            fired.append(bool(got))
    return f"found {sum(found)}/{len(found)}  fired {sum(fired)}/{len(fired)} (scenes without a held garment)"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("test", "final") or (mode == "final" and len(sys.argv) < 3):
        raise SystemExit(__doc__)
    ims = sorted((A / "images").glob("*.jpg"))
    first = min(p.stem.split("_")[0] for p in ims)
    train = [p for p in ims if mode == "final" or p.stem.startswith(first)]
    test = [p for p in ims if mode == "test" and not p.stem.startswith(first)]
    d = W / mode
    shutil.rmtree(d, ignore_errors=True)
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir()
    for p in train:
        for k in range(5):
            shutil.copy(p, d / f"images/r{k}_{p.name}")
            shutil.copy(A / "labels" / f"{p.stem}.txt", d / f"labels/r{k}_{p.stem}.txt")
    for tag, (im, lb) in SYN.items():
        for p in im.glob("*.jpg"):
            if (lb / f"{p.stem}.txt").exists():
                shutil.copy(p, d / f"images/s{tag}{p.name}")
                shutil.copy(lb / f"{p.stem}.txt", d / f"labels/s{tag}{p.stem}.txt")
    (d / "d.yaml").write_text(f"path: {d.as_posix()}\ntrain: images\nval: images\nnames:\n  0: held\n  1: arm\n  2: hanger\n")
    print(mode, "train", len(train), "real scenes x5 +", len(list((d / "images").glob("s*.jpg"))), "synthetic; test", len(test), flush=True)
    YOLO(str(BASE)).train(data=str(d / "d.yaml"), imgsz=640, epochs=8, batch=8, device="cpu", freeze=10, workers=0, val=False, plots=False,
                          verbose=False, project=str(W / "runs"), name=mode, exist_ok=True, optimizer="AdamW", lr0=.0005, warmup_epochs=1,
                          mosaic=.3, close_mosaic=2, hsv_h=.15, hsv_s=.6, hsv_v=.5, degrees=8, scale=.4, fliplr=.5)
    last = W / f"runs/{mode}/weights/last.pt"
    if mode == "test":
        print("RESULT 지금 모델", score(YOLO(str(BASE)), test))
        print("RESULT 미세학습 ", score(YOLO(str(last)), test))
    else:
        shutil.copy(last, C / f"models/{sys.argv[2]}.pt")
        print("FINAL saved", sys.argv[2], flush=True)


if __name__ == "__main__":
    main()
