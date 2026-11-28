"""more_data.py without models: tracks, filling the frames a garment was lost in, pasting"""
import random
import numpy as np
import more_data as M


def test_more_data():
    box = lambda x: (x, .3, x + .2, .8)
    held = lambda x, s=.8: [(0, s, box(x))]
    # 10 frames, 0.1 s apart: seen, lost for 3 frames (one weak box there), seen again; then a lone flicker far away
    seen = [(i * .1, held(.1 + .03 * i) if i in (0, 1, 2, 6, 7, 8) else []) for i in range(10)]
    seen[4] = (.4, [(0, .25, (.31, .32, .51, .81))])
    seen += [(3.0, held(.0, .5))]
    ks = M.tracks(seen)
    assert [[i for i, _, _ in k] for k in ks] == [[0, 1, 2, 6, 7, 8], [10]], ks
    f = M.fill(seen, ks[0])
    assert sorted(f) == [3, 4, 5]
    assert np.allclose(f[3], box(.19)) and np.allclose(f[5], box(.25))  # in between, by time
    assert f[4] == (.31, .32, .51, .81)  # the weak box sits better than the guess
    jump = [(0., held(.1)), (.1, []), (.2, [(0, .8, (.15, .1, .65, .95))])]  # either side of the gap unlike: linked, not filled
    assert len(M.tracks(jump)) == 1 and M.fill(jump, M.tracks(jump)[0]) == {}
    gap = [(0., held(.1)), (2., held(.5))]  # lost for longer than MAX_FILL: two tracks, nothing filled
    assert len(M.tracks(gap)) == 2
    rng = random.Random(0)
    bg = np.full((270, 480, 3), 90, np.uint8)
    cut = (np.full((200, 120, 3), 200, np.uint8), np.full((200, 120), 255, np.uint8))
    got = [r for r in (M.paste(bg, cut, rng) for _ in range(30)) if r]
    assert got and all(im.shape == bg.shape and m.any() and (im[m] > 100).mean() > .5 for im, m in got)


if __name__ == "__main__":
    test_more_data()
    print("ok")
