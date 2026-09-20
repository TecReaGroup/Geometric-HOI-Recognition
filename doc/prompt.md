# prompt

开始编写基于 <https://github.com/southnx/IcH-Vid-HOI> 实现 HOI 检测的代码
参考 temp/ 里面相关代码

make train:
骨架识别：rtmpose
物体特征点识别：yolo26-pose (data/model/best.pt)
输入：从 data/video/ 里面提取视频帧（需要区别正负样本），再分别进行骨架识别和特征点识别，划分数据集后训练 IcH-Vid-HOI 模型

make run：
获取摄像头视频帧，再分别进行骨架识别和特征点识别，使用训练好的 IcH-Vid-HOI 模型进行 HOI 检测，输出目标动作检测结果的置信率
