"""분류 모델 학습.

    # 공개 데이터로 종류 분류 학습 (촬영 데이터가 없어도 된다)
    python -m vision.train --task category --extra data/public

    # 촬영 데이터로 이어서 학습
    python -m vision.train --task category --init weights/category.pt --lr 1e-4

    # 방향 확인 모델 (촬영 데이터 필요)
    python -m vision.train --task view

검증 정확도가 가장 높았던 에폭의 가중치를 weights/<task>.pt 로 저장한다.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import config
from .classifier import build_model, save_checkpoint
from .dataset import GarmentDataset, scan, scan_flat, split_by_garment


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(args):
    """학습/검증 샘플을 정한다.

    촬영 데이터가 있으면 검증은 촬영 데이터로만 한다. 실제 사용 환경에서의 성능을 봐야 하기 때문이다.
    공개 데이터는 종류 라벨만 있으므로 종류 분류의 학습셋에만 보탠다.
    촬영 데이터가 아직 없으면 공개 데이터 일부를 떼어 검증에 쓴다.
    """
    own = scan(args.data)
    extra = scan_flat(args.extra) if args.extra and args.task == "category" else []

    if own:
        train_samples, val_samples = split_by_garment(own, args.val_ratio, args.seed)
        return train_samples + extra, val_samples
    if extra:
        print("촬영 데이터가 없어 공개 데이터의 일부로 검증합니다. "
              "여기서 나온 정확도는 실제 카메라 환경의 성능이 아닙니다.")
        return split_by_garment(extra, args.val_ratio, args.seed)
    raise SystemExit(f"학습할 이미지가 없습니다: {args.data}")


def class_counts(dataset):
    counts = Counter(getattr(s, dataset.task) for s in dataset.samples)
    return {c: counts.get(c, 0) for c in dataset.classes}


def class_weights(dataset):
    """장수가 적은 종류의 손실에 더 큰 가중치를 준다. 종류별 장수가 달라도 적은 쪽이 무시되지 않게 한다."""
    counts = class_counts(dataset)
    total = sum(counts.values())
    return torch.tensor([total / (len(counts) * max(counts[c], 1)) for c in dataset.classes],
                        dtype=torch.float)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """정확도와 혼동행렬(행: 정답, 열: 예측)을 계산한다."""
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()
        for t, p in zip(y, pred):
            confusion[t, p] += 1
    total = confusion.sum().item()
    acc = confusion.diag().sum().item() / total if total else 0.0
    return acc, confusion


def print_confusion(confusion, classes):
    names = [config.KOREAN.get(c, c) for c in classes]
    print("혼동행렬 (행: 정답, 열: 예측)")
    print("        " + "".join(f"{n:>6}" for n in names))
    for name, row in zip(names, confusion.tolist()):
        print(f"{name:>6}  " + "".join(f"{v:>6}" for v in row))


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_samples, val_samples = load_samples(args)
    train_set = GarmentDataset(train_samples, args.task, args.img_size, train=True)
    val_set = GarmentDataset(val_samples, args.task, args.img_size, train=False)
    classes = train_set.classes
    print(f"[{args.task}] 학습 {len(train_set)}장 {class_counts(train_set)}")
    print(f"[{args.task}] 검증 {len(val_set)}장 {class_counts(val_set)}")
    if len(val_set) == 0:
        print("검증셋이 비어 있습니다. 종류별로 옷이 2벌 이상 있어야 검증용을 떼어낼 수 있습니다.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=len(train_set) > args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained and not args.init)
    if args.init:
        # 이전 단계(공개 데이터 학습)의 가중치에서 이어서 학습한다
        ckpt = torch.load(args.init, map_location="cpu")
        if ckpt["classes"] != classes:
            raise SystemExit(f"{args.init} 의 클래스({ckpt['classes']})가 현재 설정({classes})과 다릅니다.")
        model.load_state_dict(ckpt["state_dict"])
        print(f"이어서 학습: {args.init}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_set).to(device))

    out_path = Path(args.out) / f"{args.task}.pt"
    best_acc, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += len(y)
        scheduler.step()

        val_acc, _ = evaluate(model, val_loader, device, len(classes))
        history.append({"epoch": epoch, "loss": loss_sum / seen,
                        "train_acc": correct / seen, "val_acc": val_acc})
        print(f"epoch {epoch:>2}/{args.epochs}  loss {loss_sum / seen:.4f}  "
              f"train_acc {correct / seen:.3f}  val_acc {val_acc:.3f}")

        # 검증셋이 없으면 비교할 기준이 없으므로 마지막 에폭을 저장한다
        if val_acc > best_acc or len(val_set) == 0:
            best_acc = val_acc
            save_checkpoint(out_path, model, args.arch, classes, args.img_size,
                            task=args.task, val_acc=val_acc, epoch=epoch)

    with open(out_path.with_name(f"{args.task}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if len(val_set):
        model.load_state_dict(torch.load(out_path, map_location=device)["state_dict"])
        acc, confusion = evaluate(model, val_loader, device, len(classes))
        print(f"\n최고 검증 정확도 {acc:.3f}")
        print_confusion(confusion, classes)
    print(f"저장: {out_path}")


def main():
    p = argparse.ArgumentParser(description="의류 분류 모델 학습")
    p.add_argument("--task", required=True, choices=["category", "view"])
    p.add_argument("--data", default=str(config.CROP_DIR), help="크롭된 촬영 이미지 폴더")
    p.add_argument("--extra", default=None, help="<종류>/*.jpg 구조의 공개 데이터 폴더 (category 전용)")
    p.add_argument("--init", default=None, help="이어서 학습할 가중치 파일")
    p.add_argument("--out", default=str(config.WEIGHT_DIR))
    p.add_argument("--arch", default=config.ARCH)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="ImageNet 가중치 없이 학습")
    train(p.parse_args())


if __name__ == "__main__":
    main()
