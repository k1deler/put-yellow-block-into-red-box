# 黄色物块 → 红色盒子训练

来源与许可见[主README](../README.md#代码来源与许可)，保留网络文件头和组件许可证。真实数据、模型与机器登录信息不提交仓库。

| 项目 | 设置 |
|---|---|
| 任务 | `put_block_in_bowl` |
| 网络 | ACT参考格式，双腕480×640，chunk100 |
| 动作 | 14维delta，30Hz，action_shift0 |
| 初始化 | 新任务参数＋标准ImageNet视觉骨干，不加载既有任务权重 |
| 划分 | 完整episode80/20，固定划分种子；排除失败及不完整片段 |
| 归一化 | 只用训练episode |
| 实验 | seed42、batch8、5000epoch、每500epoch保存 |
| 资源 | 空闲单GPU，独立目录且拒绝覆盖 |
| 选择 | 固定验证窗口、z=0、物理目标误差，保留候选 |

`audit.py`审计数据合同、JPEG、反馈与目标有限性、夹爪范围、episode重叠和划分。`train_one.py`先做真实数据检查训练，再进行完整实验；`select_model.py`保存候选选择结果和兼容部署checkpoint。

在已配置的训练Python环境中指定实际绝对路径：

```bash
python training/train_one.py --data /absolute/path/to/converted_data --out /absolute/path/to/new_experiment
# 前置检查后显式启动：
python training/train_one.py --data /absolute/path/to/converted_data --out /absolute/path/to/new_experiment --launch
```

数据必须为`yam_train/jpeg-v1`，包含双腕JPEG、从臂状态与时刻对齐的主臂目标。配置、数据SHA、划分和选择依据写入实验输出；服务器账户、私有路径和凭据仅保留现场。

`tests_training.py`用合成数据检查合同及一次真实反向更新，验证训练／部署网络逐元素一致，不交付随机小测试模型为任务权重。真实训练或开环误差通过仍不代表实机抓取成功，候选需要单独自检和现场验证。
