"""direction, gaps, still things (a face, a worn shirt) and crops; no models needed"""
import numpy as np
import live

F = np.zeros((72, 128, 3), np.uint8)
NEAR, FAR = (.6, 0, 1, 1), (.05, .5, .5, 1)  # at the closet / low on the user's side


def play(path, side="right", conf=.8, dt=.1):
    ev, got = live.Events(side), []
    for i, b in enumerate(path):
        got += ev.step(i * dt, [(conf, b)] if b else [], F)
    return got + ev.flush()


def way(a, z, n=12):
    return [tuple(np.array(a) + (np.array(z) - a) * i / (n - 1)) for i in range(n)]


def test_live():
    assert [m[0] for m in play(way(NEAR, FAR))] == ["out"]
    assert [m[0] for m in play(way(FAR, NEAR))] == ["in"]
    mirror = lambda b: (1 - b[2], b[1], 1 - b[0], b[3])
    assert [m[0] for m in play(way(mirror(NEAR), mirror(FAR)), "left")] == ["out"]
    assert play([(0, 0, .2, .7)] * 40) == []                                    # a face at the edge
    assert play(way(NEAR, FAR) + way(FAR, NEAR)) == []                          # out and straight back: nothing changed
    assert play(way(NEAR, FAR), conf=.5) == []                                  # never sure
    assert play(way(NEAR, FAR, 4)) == []                                        # too short
    holes = way(FAR, NEAR, 20)
    holes[5:9] = [None] * 4
    assert len(play(holes)) == 1                                                # a few lost frames: still one move
    two = play(way(NEAR, FAR) + [None] * 10 + way(FAR, NEAR))
    assert [m[0] for m in two] == ["out", "in"] and two[0][2] < two[1][1]
    action, start, end, crops = play(way(FAR, NEAR))[0]
    assert len(crops) == 5 and all(c.size for c in crops) and 0 <= start < end
    both, got = live.Events("right"), []                                        # a still face does not join the garment
    for i, b in enumerate(way(FAR, NEAR)):
        got += both.step(i * .1, [(.8, b), (.9, (0, 0, .12, .3))], F)
    assert [m[0] for m in got + both.flush()] == ["in"]
    f = [[(.1, .2), (.12, .2)]]
    assert live.is_person((0, 0, .3, .6), f) and not live.is_person((.2, 0, .6, 1), f) and not live.is_person((0, 0, .9, 1), f)


if __name__ == "__main__":
    test_live()
    print("ok")
