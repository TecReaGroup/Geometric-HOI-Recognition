# Geometric HOI Recognition

**HOI Recognition**：给定视频/序列，输出动作类别（喝水、递东西等）。常用 CAD-120、MPHOI-72/120。几何方法几乎都直接吃人体骨架 + 物体框/关键点

已有 **图片 + 人体骨架 + 物品关键点**，目标只是认特定动作：优先看 Recognition 这一支；若必须从单张图出 `<人, 动作, 物>`，再用 Detection 几何方法

---

## 对比表（几何向、可吃关键点）

| 方法           | 时间          | 任务                                   | 核心原理（非细节）                                           | 准确率                                                       | 性能                     | 是否匹配你的输入                                       | 链接                                                         |
| -------------- | ------------- | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------ | ------------------------------------------------------------ |
| **PMFNet**     | ICCV 2019     | 图像 Detection                         | 人体姿态当空间先验，和人/物外观、相对布局一起判断交互        | V-COCO AP_role **52.0**                                      | 两阶段，偏慢             | 图+人体关键点；物体多为框不是物体关键点                | [GitHub](https://github.com/bobwan1995/PMFNet)               |
| **SGCN4HOI**   | IEEE SMC 2022 | 图像 Detection                         | 人体关键点 + 物体骨架关键点建一张图，用 GCN 学人–物结构接触关系，再乘上外观分数 | V-COCO **53.1 / 57.9** (S1/S2)                               | 两阶段，轻于 Transformer | **高度匹配**：人 17 点 + 物 9 点                       | [GitHub](https://github.com/zhumanli/SGCN4HOI) · [论文](https://arxiv.org/abs/2207.05733) |
| **2G-GCN**     | ECCV 2022     | **视频 Recognition**                   | 下层图学人体骨架与物体几何的空间依赖，上层图再和外观融合     | CAD-120 子活动 F1@10 **89.5**；MPHOI-72 F1@10 **68.6**       | ~148M，约 **84 ms/帧**   | **高度匹配**：骨架位置/速度 + 物体对角点               | [GitHub](https://github.com/tanqiu98/2G-GCN) · [论文](https://arxiv.org/abs/2207.09425) |
| **HOKEM**      | ICIP 2023     | 图像 Detection 插件                    | 从分割抽出物体关键点，和人体关键点进自适应 GCN，输出与外观模型相乘 | 外观基线 49.7 → **+HOKEM 54.6** (V-COCO S1，约 +5 mAP)       | 插件，额外一次小 GCN     | **高度匹配**：人 17 点 + 物 9 点                       | [论文](https://arxiv.org/abs/2306.14260)（未见官方代码）     |
| **CATS**       | arXiv 2024    | **视频 Recognition**                   | 先按“人类别 / 物类别”各自做几何图，再融外观，最后在场景图上学类别间关系 | MPHOI-72 F1@10 **71.3**（高于 2G-GCN）                       | ~132M，约 **137 ms/帧**  | 匹配：人体关节 + 物体对角点 + 视频外观                 | [论文](https://arxiv.org/abs/2407.00917)                     |
| **GeoHOI**     | 2024          | 图像 Detection                         | 用 UniPointNet 给任意物体学统一关键点；关键点相似度估“是否在交互”，局部关键点块增强 Transformer query | V-COCO **67.8 / 73.3**（增强版 69.4 / 74.4）；HICO-DET Full **30.07**（增强版 35.05） | 约 **589 ms/图**         | 图+分割/关键点；可接你已有关键点，但原论文自学生物体点 | [GitHub](https://github.com/zhumanli/GeoHOI) · [UniPointNet](https://github.com/zhumanli/UniPointNet) · [论文](https://arxiv.org/abs/2406.18691) |
| **GeoVis-GNN** | ESWA 2025     | **视频 Recognition（当前几何线最强）** | 先在实体级用双注意力融几何与外观，再在实体互依赖图上建模交互（自底向上） | MPHOI-72 F1@10 **84.3**；MPHOI-120 F1@10 **65.1**（均超 CATS/2G-GCN） | ~130M，约 **119 ms/帧**  | **最匹配**：骨架 + 物体几何 + 可选 RGB                 | [GitHub](https://github.com/tanqiu98/GeoVis-GNN) · [论文](https://arxiv.org/abs/2506.03440) |

数字来自原论文报告，跨数据集不可直接比：Recognition 用 F1@overlap，Detection 用 mAP。
