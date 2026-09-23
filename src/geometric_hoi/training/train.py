"""Train and select official-model binary action checkpoints."""

import logging
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..action.feature import FeatureExtractor
from ..action.model import ActionModel
from ..recognition.engine import fingerprint as weight_digest
from ..setting import ROOT, checkpoint_path
from .dataset import build_dataset

LOGGER = logging.getLogger(__name__)


def train(setting: dict) -> None:
    """Train the selected official network and save the best validation checkpoint."""
    option = setting["train"]
    random.seed(option["seed"])
    np.random.seed(option["seed"])
    torch.manual_seed(option["seed"])
    device = torch.device(setting["model"]["device"])
    fingerprint = weight_digest(ROOT / setting["recognition"]["object"]["object_weight"])
    extractor = FeatureExtractor(setting)
    training, validation = build_dataset(setting, extractor, fingerprint)
    point_count = extractor.object_point_count
    del extractor
    network = ActionModel(setting, point_count).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=option["learning_rate"])
    counts = np.bincount([entry["label"] for entry in training.entries], minlength=2)
    class_weight = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.NLLLoss(weight=class_weight)
    training_loader = DataLoader(training, batch_size=option["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation, batch_size=option["batch_size"])
    best_loss = float("inf")
    target = checkpoint_path(setting)
    target.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, option["epochs"] + 1):
        network.train()
        total_loss = 0.0
        for human, object_feature, label in training_loader:
            optimizer.zero_grad(set_to_none=True)
            probability = network(human.to(device), object_feature.to(device))
            loss = criterion(probability, label.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; inspect feature quality")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(label)
        network.eval()
        validation_loss, correct = 0.0, 0
        with torch.inference_mode():
            for human, object_feature, label in validation_loader:
                label = label.to(device)
                probability = network(human.to(device), object_feature.to(device))
                validation_loss += nn.functional.nll_loss(probability, label, reduction="sum").item()
                correct += int((probability.argmax(dim=1) == label).sum())
        validation_loss /= len(validation)
        if not np.isfinite(validation_loss):
            raise RuntimeError("Nonfinite validation loss")
        LOGGER.info("model=%s epoch=%d/%d train_loss=%.5f val_loss=%.5f val_accuracy=%.4f",
                    setting["model"]["name"], epoch, option["epochs"], total_loss / len(training),
                    validation_loss, correct / len(validation))
        if validation_loss < best_loss:
            best_loss = validation_loss
            temporary = ROOT / "temp" / f"{target.stem}.partial.pt"
            torch.save({"setting": setting,
                        "state_dict": network.state_dict(), "object_point_count": point_count,
                        "object_weight_digest": fingerprint, "epoch": epoch,
                        "validation_loss": best_loss,
                        "validation_accuracy": correct / len(validation)}, temporary)
            temporary.replace(target)
            LOGGER.info("Saved best checkpoint %s", target)
