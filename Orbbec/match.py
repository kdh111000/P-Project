"""Is the garment in the hand one of the registered ones? Two fine-tuned encoders (fashionSigLIP-B, DINOv2-L) plus colour
compare the frames of one put/take with 3 photos (front, back, side) per garment; a binary classifier turns that into
p(registered). Weights: match/match_siglip.pt, match/match_dino.pt, match/decider.npz (classifier, unrelated garments)."""
from functools import lru_cache
from pathlib import Path
import math
import cv2
import numpy as np

W = Path(__file__).resolve().parent / "match"
MARGIN = 0.075   # |p - 0.5| below this -> unsure; 1.5 % of the public test. ponytail: refit on real door events
PAD = 50         # the closet is filled up to this with unrelated garments: in a closet of 5 the classifier has too little to compare with
MODELS = ("siglip", "dino")
WEIGHT = (1, 1, .3)  # siglip, dino, colour: how the best garment is picked


def square(im, size=224, bg=255):
    h, w = im.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(im, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), bg, np.uint8)
    y, x = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
    out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


@lru_cache(None)
def nets():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import open_clip
    from transformers import Dinov2Config, Dinov2Model
    out = []
    for name, m in (("siglip", open_clip.create_model("ViT-B-16-SigLIP", pretrained=None).visual),
                    ("dino", Dinov2Model(Dinov2Config(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, image_size=518, patch_size=14)))):
        want = m.state_dict()
        s = {}
        for k, v in torch.load(W / f"match_{name}.pt", map_location="cpu").items():
            if k not in want:  # transformers renamed DINOv2's attention layers between versions
                k = (k.replace(".attention.q_proj", ".attention.attention.query").replace(".attention.k_proj", ".attention.attention.key")
                      .replace(".attention.v_proj", ".attention.attention.value").replace(".attention.o_proj", ".attention.output.dense"))
            s[k] = v.float()
        m.load_state_dict(s)
        mean, sd = ([.5] * 3,) * 2 if name == "siglip" else ([.485, .456, .406], [.229, .224, .225])
        out.append((name, m.eval().to(dev), torch.tensor(mean, device=dev).view(1, 3, 1, 1), torch.tensor(sd, device=dev).view(1, 3, 1, 1), dev))
    return out


def colour(ims):
    out = []
    for im in ims:
        h = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_RGB2HSV)], [0, 1, 2], (im.min(2) < 250).astype(np.uint8), [12, 4, 4], [0, 180, 0, 256, 0, 256])
        out.append((h / (h.sum() + 1e-6)).flatten())
    out = np.sqrt(np.array(out))
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def embed(ims):
    """BGR images -> [siglip (n, 768), dino (n, 1024), colour (n, 192)], unit length"""
    import torch
    import torch.nn.functional as F
    ims = [square(im)[..., ::-1].copy() for im in ims]
    out = []
    for name, m, mean, sd, dev in nets():
        v = []
        for i in range(0, len(ims), 8):
            with torch.inference_mode(), torch.autocast(dev, enabled=dev == "cuda"):
                x = (torch.from_numpy(np.stack(ims[i:i + 8])).to(dev).permute(0, 3, 1, 2).float() / 255 - mean) / sd
                o = m(x) if name == "siglip" else m(pixel_values=x).last_hidden_state[:, 0]
            v.append(F.normalize(o.float(), dim=-1).cpu().numpy())
        out.append(np.concatenate(v))
    return out + [colour(ims).astype(np.float32)]


@lru_cache(None)
def decider():
    return dict(np.load(W / "decider.npz"))


def is_garment(e):
    """embed() of crops -> p of each that it shows a garment rather than a face, an arm or the user's body"""
    D = decider()
    return 1 / (1 + np.exp(-(((np.hstack(e[:2]) - D["gate_mean"]) / D["gate_scale"]) @ D["gate_coef"] + D["gate_bias"])))


def decide(query, closet, same=None):
    """query: embed() of the frames of one put/take; closet: {key: embed() of its photos}
    -> (key or None, p that it is in the closet, [the 3 most alike keys]). None: an unrelated garment is more alike.
    same: {key: keys that look the same}; they are one garment to the classifier (which of them it is, the state tells)"""
    D, keys = decider(), list(closet)
    n_pad = max(0, PAD - len(keys))
    views = lambda e: np.asarray(e)[[0, len(e) // 2, -1]] if len(e) != 3 else np.asarray(e)  # 3 per garment, whatever was stored
    N, T = len(keys) + n_pad, []
    for i, w in enumerate((*MODELS, "colour")):
        G = np.stack([views(closet[k][i]) for k in keys] + list(D[f"pad_{w}"][:n_pad].astype(np.float32)))  # N, 3, dim
        C, Q = D[f"c_{w}"].astype(np.float32), query[i]
        K = min(50, len(C))
        s = np.einsum("fd,nvd->fnv", Q, G).max(2)  # frames x garments
        top = np.sort((Q @ C.T).mean(0))[-K:]
        cg = np.sort(G.reshape(-1, G.shape[2]) @ C.T, 1)[:, -K:].reshape(N, 3, K).mean(1)
        m = s.mean(0)
        T.append(dict(S=m, X=s.max(0), D=s.std(0), q=top.mean(),
                      Z=((m - top.mean()) / (top.std() + 1e-6) + (m - cg.mean(1)) / (cg.std(1) + 1e-6)) / 2))
    f = sum(w * x["Z"] for w, x in zip(WEIGHT, T))
    o = np.argsort(-f)
    b = o[0]
    rest = np.array([j for j in o[1:] if not (b < len(keys) and j < len(keys) and keys[j] in (same or {}).get(keys[b], ()))])
    v = [f[b], f[b] - f[rest[0]], math.log(N)]
    for x in T:
        v += [x["S"][b], x["X"][b], x["D"][b], x["Z"][b], x["q"], *(lambda s: (s[-1] - s[-2], float(s[-1] == x["S"][b])))(np.sort(x["S"][[b, *rest]]))]
    p = 1 / (1 + math.exp(-float(((np.array(v) - D["mean"]) / D["scale"]) @ D["coef"] + D["bias"])))
    return (keys[b] if b < len(keys) else None), p, [keys[j] for j in o if j < len(keys)][:3]


def twins(closet, margin=MARGIN):
    """{key: the keys whose garments the model cannot tell from it (the same garment bought twice)}"""
    same = {k: {k} for k in closet}
    for k, e in closet.items():
        other = {j: v for j, v in closet.items() if j != k}
        best, p, _ = decide(e, other) if other else (None, 0, [])
        if best is not None and p >= .5 + margin:
            g = same[k] | same[best]
            for j in g:
                same[j] = g
    return same


def cut_out(bgr):
    """registration photo -> the garment on white (rembg), like the photos the encoders were trained on; as is if nothing was cut"""
    from rembg import new_session, remove
    if not hasattr(cut_out, "s"):
        cut_out.s = new_session("u2net")
    k = remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), session=cut_out.s, only_mask=True) >= 128
    if k.mean() < .02:
        return bgr
    ys, xs = np.where(k)
    return np.where(k[..., None], bgr, 255).astype(np.uint8)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
