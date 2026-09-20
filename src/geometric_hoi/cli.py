"""Command-line entry points for setup, training and camera inference."""

import argparse
import logging
from pathlib import Path

from .logging import configure_logging
from .setting import ROOT, configure_directory, load_setting


def main() -> None:
    """Dispatch a complete application operation."""
    parser = argparse.ArgumentParser(description="Official 2G-GCN / GeoVis-GNN HOI recognition")
    parser.add_argument("command", choices=("prepare", "train", "run"))
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "config.toml")
    arguments = parser.parse_args()
    configure_directory()
    configure_logging()
    try:
        setting = load_setting(arguments.config)
        if arguments.command in {"train", "run"}:
            import torch

            device = torch.device(setting["model"]["device"])
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; run make install and check the NVIDIA driver")
                torch.cuda.set_device(device)
                logging.getLogger(__name__).info(
                    "PyTorch=%s CUDA=%s device=%s GPU=%s", torch.__version__,
                    torch.version.cuda, device, torch.cuda.get_device_name(device),
                )
        if arguments.command == "prepare":
            from .upstream import REVISION, prepare_source

            for name in REVISION:
                prepare_source(name)
        elif arguments.command == "train":
            from .train import train

            train(setting)
        else:
            from .runtime import run_camera

            run_camera(setting)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped by user")
    except Exception:
        logging.getLogger(__name__).exception("%s failed", arguments.command)
        raise SystemExit(1) from None
