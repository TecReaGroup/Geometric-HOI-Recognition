# Plan

human: yolo26m
skeleton: rtmlib balanced
HOI: geovis-gnn
keypoint: yolo26-pose

## yolo26-pose 元数据

names:
  0: pen

kpt_shape: [2, 3]

kpt_names:
  0:
    - tip
    - tail

kpt_shape [2, 3] 表示两个关键点，和每个关键点维度： x、y、可见性/置信度
