"""Adapt official networks to single-person RGB keypoint clips."""

import math
from typing import Any

import torch
from torch import nn

from .upstream import official_class

JOINT_COUNT = 17
GEOMETRY_SIZE = (JOINT_COUNT + 2) * 4


class ActionModel(nn.Module):
    """Train official stage-one recognition with video-level binary labels."""

    def __init__(self, setting: dict, object_point_count: int) -> None:
        """Construct all parameters before the optimizer and checkpoint loading."""
        super().__init__()
        name = setting["model"]["name"]
        window = setting["train"]["window_frames"]
        arguments: dict[str, Any] = dict(
            input_size=(2048 + GEOMETRY_SIZE, 2048 + object_point_count * 3),
            num_classes=(2, None), hidden_size=setting["model"]["hidden_size"],
            message_humans_to_human=False, message_objects_to_object=False,
            message_segment=False, message_type="v2", message_granularity="v1",
            attention_style="v3",
        )
        if name == "2g-gcn":
            arguments.update(gcn_node=JOINT_COUNT + 2, message_geometry_to_human=True)
        else:
            arguments.update(no_human=1, no_human_joints=JOINT_COUNT, max_no_objects=1)
        self.network = official_class(name)(**arguments)
        if name == "geovis-gnn":
            fusion = self.network.masked_GCN.fuse_conv
            fusion.seq_length = window
            fusion.temporal_fusion = nn.Sequential(nn.Conv2d(window, window, 1), nn.GELU())

    def forward(self, human: torch.Tensor, object_feature: torch.Tensor) -> torch.Tensor:
        """Return clip log probabilities from stage-one segment recognition."""
        batch, steps = human.shape[:2]
        # Official stage one imposes updates at every frame; no boundary labels are available.
        boundary = human.new_ones((batch, steps, 1))
        presence = object_feature.abs().sum(dim=(1, 3)).gt(0).to(human.dtype)
        output = self.network(human, object_feature, presence,
                              human_segmentation=boundary, objects_segmentation=boundary)
        frame_log_probability = output[4][:, :, :, 0]
        return torch.logsumexp(frame_log_probability, dim=-1) - math.log(steps)
