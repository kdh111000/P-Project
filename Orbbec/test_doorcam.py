import json, tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import doorcam
import requests as doorcam_requests


def test_doorcam():
    shut = np.zeros((48, 64), np.uint16)
    room = np.full((48, 64), 3000, np.uint16)
    def someone(mm):
        d = room.copy()
        d[10:40, 20:44] = mm
        return d
    rgb = lambda v: np.full((48, 64, 3), 100 + v, np.uint8)
    seq = ([shut] * 3 + [room, someone(1000), someone(900)] + [shut] * 2  # hand over the lens: same session
           + [someone(800), room] + [shut] * 6 + [room] * 2 + [shut] * 6)   # 2nd opening: nobody came
    frames = [(1_800_000_000 + i, rgb(i) if d.any() else rgb(-90), d) for i, d in enumerate(seq)]  # shut = dark
    assert not doorcam.door_open(room, rgb(-90))  # lens covered / door shut in the dark
    got = list(doorcam.sessions(frames, room))  # bg = empty room

    with tempfile.TemporaryDirectory() as tmp:  # recording: every STEP-th door-open frame, then the sharpest person frames
        tape = doorcam.Tape(Path(tmp))
        noisy = np.random.default_rng(0).integers(0, 255, (48, 64, 3), np.uint8)
        blur = [(t, (cv2.GaussianBlur(noisy, (15, 15), 0) if i % 2 else noisy.copy()) if c.mean() > 35 else c, d) for i, (t, c, d) in enumerate(frames)]
        assert len(list(doorcam.sessions(blur, room, tape))) == 2
        d = doorcam.folder(1_800_000_003, Path(tmp))
        assert len(list((d / "depth").glob("*.png"))) == 2  # 5 open frames (4 + 1 after the hand) -> frames 0 and 3
        before, after = doorcam.best_frames(d, room)
        assert (before[1] < 3000).any() and after is not None
    assert len(got) == 2, len(got)
    assert (doorcam.background(iter(frames[3:4] * 30)) == room).all()
    (o, c, before, after), second = got
    assert (o, c) == (1_800_000_003, 1_800_000_014)
    assert before[0][0, 0, 0] == 104 and after[0][0, 0, 0] == 108 and second[2] is None

    with tempfile.TemporaryDirectory() as tmp:
        d = doorcam.save(o, c, before, after, "a" * 16, Path(tmp))
        assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["person"] is True
        assert cv2.imread(str(d / "after_depth.png"), cv2.IMREAD_UNCHANGED)[20, 30] == 800

        # judge: Qwen's numbers -> garment keys, both-ways and unknown numbers dropped, "out" flags flipped
        store = Path(tmp) / "uploads"
        for i, (cat, out) in enumerate((("아우터", False), ("하의", True), ("상의", False)), 1):
            g = store / cat / f"g{i}"
            g.mkdir(parents=True)
            Image.fromarray(rgb(i)).save(g / "front_cut.png")  # cv2.imwrite fails silently on Korean paths
            (g / "meta.json").write_text(json.dumps({"time": str(i), "closet": "a" * 16, "in_closet": True, "category": cat, "out": out}), encoding="utf-8")
        seen = []
        def ask(images, prompt, labels, side, max_tokens, grammar):
            seen.append(labels)  # cut colour tells which shot it came from: before -> garment 2, after -> garment 1
            return {"desc": "x", "match": 2 if images[0].getpixel((0, 0))[0] == 105 else 1}
        cut = lambda v: {"label": "jacket", "score": .3, "box": [0, 0, 9, 9], "mask": np.zeros((48, 64), bool), "cut": Image.fromarray(rgb(v))}
        shots = iter([[cut(5)], [cut(6), cut(7)]])  # opening: garment 2 in hand; closing: garment 1 and an unknown piece
        m = doorcam.judge(d, "a" * 16, ask, store, find=lambda im, near: next(shots))
        assert seen[0][0] == "[조각]" and len(seen[0]) == 4 and len(seen) == 3
        assert (m["taken"], m["returned"]) == (["아우터/g1"], ["하의/g2"]), m
        assert m["found_after"][0]["garment"] == "아우터/g1" and (d / "after_seg.jpg").exists()
        flag = lambda k: json.loads((store / k / "meta.json").read_text(encoding="utf-8"))["out"]
        assert flag("아우터/g1") is True and flag("하의/g2") is False and flag("상의/g3") is False
        def down(*a, **k):
            raise doorcam_requests.ConnectionError("server down")
        assert "error" in doorcam.judge(d, "a" * 16, down, store, find=lambda im, near: [cut(5)])


if __name__ == "__main__":
    test_doorcam()
    print("ok")
