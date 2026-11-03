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

选模同时考虑右臂、右夹爪的完整chunk误差和首步误差：先排除所有指标均不优的候选，再最小化相对各指标最佳值的最坏误差比。`selection_quality.json`保留指标、参考值及排名。夹爪误差不再只用于右臂误差相等时的比较。该规则仅选择离线候选，不改当前部署权重；现场收益仍须验证。

全量训练候选没有独立验证集，`select_model.py`默认拒绝选择；仅显式传入`--allow-training-overlap`时使用检查点记录的`diagnostic_reference_episodes`，输出中保留每个候选的重叠名单和训练内诊断标签，不能将其称作泛化验证。所有候选必须使用完全相同的参考episode。`--filename`可指定导出文件名。

`fine_tune.py`从已有本任务检查点在新目录中继续优化，保留相机、动作、时间对齐和父模型归一化；原导出未保存优化器，因此明确重置AdamW，不能声称精确恢复旧训练。默认保留原训练／验证划分；显式`--scope all`可训练全部已审计episode，但原验证集会成为训练内诊断，报告必须标记重叠，不能声称未见数据上的提升。新目录保存父权重SHA、配置、训练日志、兼容部署候选和单独的优化器／随机状态；已有部署不会自动切换。仅前置检查时不传`--launch`；开始训练须显式指定该参数。GPU任务须由外层资源检查和锁控制，不中断他人进程。

在已配置的训练Python环境中指定实际绝对路径：

```bash
python training/train_one.py --data /absolute/path/to/converted_data --out /absolute/path/to/new_experiment
# 前置检查后显式启动：
python training/train_one.py --data /absolute/path/to/converted_data --out /absolute/path/to/new_experiment --launch
```

数据必须为`yam_train/jpeg-v1`，包含双腕JPEG、从臂状态与时刻对齐的主臂目标。配置、数据SHA、划分和选择依据写入实验输出；服务器账户、私有路径和凭据仅保留现场。

`tests_training.py`用合成数据检查合同及一次真实反向更新，验证训练／部署网络逐元素一致，不交付随机小测试模型为任务权重。真实训练或开环误差通过仍不代表实机抓取成功，候选需要单独自检和现场验证。
