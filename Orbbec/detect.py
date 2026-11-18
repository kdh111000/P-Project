"""Garment finder for doorcam frames. OWLv2 finds garments from text (worn, held in a hand or hanging),
SAM cuts each one out, depth keeps the ones in front of the empty scene (with the user, not lying on the bed),
FashionCLIP shortlists which registered garments each cut-out could be.
Bench on saved frames (CPU): OWLv2 3 s + SAM ~4 s per frame; SegFormer-clothes merged a held garment into the shirt,
Grounding DINO missed a motion-blurred held garment, OWLv2 caught both."""
from functools import lru_cache
import numpy as np
import torch
from PIL import Image, ImageDraw

OWL, SAM = "google/owlv2-base-patch16-ensemble", "facebook/sam-vit-base"
# the last three catch what a closet camera sees most: a camouflage jacket on a hanger scored 0.07 as "clothes",
# 0.20 as "garment on a hanger"; a held pair of shorts 0.26 -> 0.41 as "clothes held in a hand"
LABELS = ["clothes", "jacket", "coat", "shirt", "t-shirt", "hoodie", "pants", "shorts", "skirt", "dress",
          "garment on a hanger", "clothes held in a hand", "folded clothes"]
MIN_SCORE = 0.05  # ponytail: from ~12 bench frames; the camouflage jacket's whole-garment box scored 0.07
MIN_AREA = 0.04   # share of the frame: at 0.2 "garment on a hanger" boxes were 0.1% slivers of the hanger hook,
                  # which Qwen then "recognised" as the first garment offered
NEAR_FRAC = 0.3   # share of a garment's pixels in front of the empty scene to count as "with the user"
MAX_CUTS = 3      # per frame: Qwen's 4096-token context holds ~20 images


@lru_cache(None)
def models():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor
    return (Owlv2Processor.from_pretrained(OWL), Owlv2ForObjectDetection.from_pretrained(OWL).eval(),
            SamProcessor.from_pretrained(SAM), SamModel.from_pretrained(SAM).eval())


@torch.no_grad()
def find(im, near=None):
    """PIL image (+ optional bool mask of pixels in front of the empty scene) ->
    [{"box", "label", "score", "mask", "cut"}] best first, at most MAX_CUTS, only garments with the user when near is given"""
    from torchvision.ops import nms
    op, om, sp, sm = models()
    x = op(text=[["a photo of " + l for l in LABELS]], images=im, return_tensors="pt")
    side = max(im.size)  # OWLv2 pads to a square
    r = op.post_process_grounded_object_detection(om(**x), threshold=MIN_SCORE, target_sizes=torch.tensor([[side, side]]))[0]
    keep = nms(r["boxes"], r["scores"], 0.4).tolist()
    if not keep:
        return []
    boxes = [[min(max(0.0, v), lim) for v, lim in zip(r["boxes"][i].tolist(), im.size * 2)] for i in keep]
    s = sp(im, input_boxes=[boxes], return_tensors="pt")
    pred = sm(**s, multimask_output=False).pred_masks
    masks = sp.image_processor.post_process_masks(pred, s["original_sizes"], s["reshaped_input_sizes"])[0][:, 0].numpy()
    a = np.array(im.convert("RGB"))
    out = []
    for i, b, mk in zip(keep, boxes, masks):
        if mk.mean() < MIN_AREA or (near is not None and near[mk].mean() < NEAR_FRAC):
            continue
        x0, y0, x1, y1 = map(int, b)
        cut = np.where(mk[..., None], a, 255).astype(np.uint8)[y0:y1 + 1, x0:x1 + 1]
        if cut.size:
            out.append({"box": b, "label": LABELS[int(r["labels"][i])], "score": float(r["scores"][i]), "mask": mk, "cut": Image.fromarray(cut)})
    return out[:MAX_CUTS]


def overlay(im, found):
    a = np.array(im.convert("RGB"))
    for i, g in enumerate(found):
        c = np.array([(255, 60, 60), (60, 255, 60), (60, 140, 255)][i % 3])
        a[g["mask"]] = a[g["mask"]] // 2 + c // 2
    im = Image.fromarray(a)
    dr = ImageDraw.Draw(im)
    for i, g in enumerate(found, 1):
        dr.rectangle(g["box"], outline=(255, 255, 0), width=3)
        dr.text((g["box"][0] + 4, g["box"][1] + 4), f"{i} {g['label']} {g['score']:.2f}", fill=(255, 255, 0))
    return im


def shortlist(cuts, garments, k=5):
    """cuts: PIL images, garments: [(key, PIL front cut-out)] -> the k garments most like any cut (FashionCLIP).
    Similarity alone can't decide: same garment 0.46-0.91, different garments 0.54-0.70 on our uploads."""
    if len(garments) <= k:
        return garments
    if not cuts:
        return garments[-k:]  # newest
    import retrieve
    sims = (retrieve.embed(cuts) @ retrieve.embed([g for _, g in garments]).T).max(0)
    return [garments[i] for i in np.argsort(-sims)[:k]]
