"""Extract frozen backbone features for CIFAR-100 and save them to disk."""
import argparse
import os

import numpy as np
import torch
import yaml
from torchvision import models, transforms
from tqdm import tqdm

from kfoc.data import load_cifar100
from kfoc.seed import set_seed


def build_backbone(name: str):
    """Return an ImageNet-pretrained torchvision backbone without its classification head."""
    if name != "resnet18":
        raise ValueError(f"Unsupported backbone: {name}")
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    return net.eval()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar100.yaml")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    d = cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_backbone(d["backbone"]).to(device)
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    os.makedirs(d["features_dir"], exist_ok=True)
    for split in ["train", "test"]:
        ds = load_cifar100(d["hf_dataset"], split)
        feats, labels = [], []
        for i in tqdm(range(0, len(ds), args.batch_size), desc=split):
            batch = ds[i : i + args.batch_size]
            x = torch.stack([tf(img.convert("RGB")) for img in batch["img"]]).to(device)
            feats.append(net(x).cpu().numpy())
            labels.append(np.asarray(batch[d["label_column"]]))
        np.savez(os.path.join(d["features_dir"], f"{split}.npz"),
                 x=np.concatenate(feats), y=np.concatenate(labels))


if __name__ == "__main__":
    main()
