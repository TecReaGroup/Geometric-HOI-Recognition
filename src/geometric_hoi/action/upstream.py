"""Pinned official source preparation with explicit compatibility patches."""

import importlib
import logging
import subprocess
import sys
from pathlib import Path

from ..setting import ROOT

LOGGER = logging.getLogger(__name__)
REVISION = {
    "2g-gcn": ("https://github.com/tanqiu98/2G-GCN.git", "02317ba47f58fc22a2f8906b305cd7595dc606e6"),
    "geovis-gnn": ("https://github.com/tanqiu98/GeoVis-GNN.git", "839cc6462ea43bcc3249678fb8c2b6b7c6920c19"),
}
PATCH_VERSION = 3
SOURCE_OPTIMIZATION_VERSION = 1


def prepare_source(name: str) -> Path:
    """Build an isolated import namespace from a pinned official Git revision."""
    url, revision = REVISION[name]
    namespace = "official_" + name.replace("-", "_")
    destination = ROOT / "temp" / "vendor" / namespace
    marker = destination / ".revision"
    signature = f"{revision}:{PATCH_VERSION}:{SOURCE_OPTIMIZATION_VERSION}"
    if marker.exists() and marker.read_text() == signature:
        return destination
    checkout = ROOT / "temp" / "source" / name
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if not checkout.exists():
        subprocess.run(["git", "clone", url, str(checkout)], check=True)
    subprocess.run(["git", "fetch", "origin", revision], cwd=checkout, check=True)
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision], cwd=checkout, text=True
    ).splitlines()
    destination.mkdir(parents=True, exist_ok=True)
    for relative in tracked:
        if not (relative.endswith(".py") and relative.startswith(("vhoi/", "pyrutils/"))
                or relative == "LICENSE"):
            continue
        source = subprocess.check_output(
            ["git", "show", f"{revision}:{relative}"], cwd=checkout
        ).decode("utf-8")
        if relative.endswith(".py"):
            source = source.replace("from pyrutils", f"from {namespace}.pyrutils")
            source = source.replace("import pyrutils", f"import {namespace}.pyrutils")
        if relative == "vhoi/models.py":
            # Select the next segment end on-device; trailing frames keep their own state.
            start = source.index("    batch_size = hx_s.size(0)", source.index("def reorder_hidden_states("))
            end = source.index("    return hx_s", start) + len("    return hx_s")
            source = source[:start] + (
                "    steps = hx_s.size(1)\n"
                "    positions = torch.arange(steps, device=hx_s.device).expand_as(ux_s)\n"
                "    ends = torch.where(ux_s != 0, positions, steps)\n"
                "    following = ends.flip(1).cummin(dim=1).values.flip(1)\n"
                "    indexes = torch.where(following < steps, following, positions)\n"
                "    return hx_s.gather(1, indexes.unsqueeze(-1).expand_as(hx_s))"
            ) + source[end:]
            # An all-missing object row must not create NaN gradients in softmax.
            source = source.replace("fill_value=float('-inf')", "fill_value=torch.finfo(att_weights.dtype).min")
            source = source.replace("torch.full_like(distances, fill_value=torch.finfo(att_weights.dtype).min)",
                                    "torch.full_like(distances, fill_value=torch.finfo(distances.dtype).min)")
            start = source.index("        if x_human.shape[3] == 2124:")
            end = source.index("        bs, t, vw", start) if name == "2g-gcn" else source.index("        # visual features", start)
            source = source[:start] + (
                "        x_geometry = x_human[:, :, 0, 2048:]\n"
                "        x_human = x_human[:, :, :, :2048]\n"
            ) + source[end:]
            if name == "2g-gcn":
                old = "x_geometry = x_geometry.unsqueeze(2)\n        x_geometry = x_geometry.view(bs, t, 1, x_geometry.shape[1] * x_geometry.shape[3])"
                source = source.replace(old, "x_geometry = x_geometry.permute(0, 3, 2, 1).contiguous().view(bs, t, 1, -1)")
            else:
                source = source.replace("num_steps=10", "num_steps = x_human.size(1)")
                source = source.replace("return batch_wise_edges_stack.to(device)", "return batch_wise_edges_stack.to(human_geometry_feature.device)")
                start = source.index("        batch_size = human_joints.shape[0]", source.index("    def get_valid_frame"))
                end = source.index("    def get_graph", start)
                source = source[:start] + (
                    "        visible = human_joints.abs().sum(dim=-1) > 0\n"
                    "        positions = torch.arange(1, human_joints.size(1) + 1, device=human_joints.device)\n"
                    "        lengths = (visible * positions).amax(dim=1).tolist()\n"
                    "        return visible.unsqueeze(-1).expand_as(human_joints), lengths\n\n"
                ) + source[end:]
        if relative == "pyrutils/torch/models_2newgat.py":
            # Preserve the singleton batch axis and mask each missing geometric frame.
            source = source.replace("mask[:valid_step_index[batch], :, :] = 1",
                                    "mask[:] = (x[batch].abs().sum(dim=-1, keepdim=True) > 0)")
            source = source.replace(".unsqueeze(0)).squeeze()", ".unsqueeze(0)).squeeze(0)")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    (destination / "__init__.py").write_text("", encoding="utf-8")
    marker.write_text(signature, encoding="utf-8")
    LOGGER.info("Prepared official source %s revision=%s patch=%d optimization=%d",
                name, revision, PATCH_VERSION, SOURCE_OPTIMIZATION_VERSION)
    return destination


def official_class(name: str) -> type:
    """Import the selected official model without colliding package names."""
    destination = prepare_source(name)
    sys.path.insert(0, str(destination.parent))
    module = importlib.import_module(f"{destination.name}.vhoi.models")
    return getattr(module, "TGGCN" if name == "2g-gcn" else "GeoVisGNN")
