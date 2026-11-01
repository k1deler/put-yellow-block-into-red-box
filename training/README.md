# 新的一组黄块／红盒子 ACT 权重

目标是按已跑通的ACT结构开展**真实的新训练**，保持`yam_train/act_ref-v1`部署合同；不是改名现有权重，也不是把两份不兼容模型做平均。训练尚未在服务器启动，不能把本目录当成已经训练好的模型。

## 来源和新实验

训练源：[FuFutheBest/yam_train](https://github.com/FuFutheBest/yam_train)，固定提交见`source.json`。`upstream/`保留训练器、数据格式、网络、来源及许可证；未复制机器登录凭据。

| 项目 | 本次设置 |
|---|---|
| 任务 | `put_block_in_bowl`，黄色物块放红容器 |
| 网络 | ACT参考实现，ResNet18＋Transformer，hidden512，encoder4、decoder7，chunk100 |
| 初始化 | 新ACT任务参数＋标准ImageNet视觉骨干；**不加载fufu的已训练任务权重** |
| 图像 | 右腕、左腕；原始JPEG；480×640；30Hz |
| 动作 | 14维delta，包括夹爪；action_shift0；沿用训练源码合同 |
| 数据 | 完整episode80/20训练／验证，split_seed1；失败episode_0024拒绝混入 |
| 归一化 | 只用训练episode拟合，区别于上游默认`all` |
| 新随机种子 | 42 |
| 优化 | 原ACT AdamW，lr和backbone_lr均1e-5，batch8，KL权重10 |
| 首组训练 | 5000epoch，约25000更新（39训练episode时）；每500epoch保存 |
| 资源 | 单张空闲GPU，启动时内存占用<1000MiB、利用率<5%；不终止其他人的任务 |
| 时限 | 真实数据smoke最多15分钟；完整训练最多6小时；评估最多1小时。到时标记未完成，不伪称训练成功 |
| 选择 | 固定512个验证窗口、z=0、物理目标误差；不使用随机后验val loss最小值直接当最佳部署模型 |
| 输出 | 独立输出目录，拒绝覆盖；新权重、代码／数据SHA、配置、划分、日志及选择报告 |

本地CPU合成数据测试只检查数据审计和一次真实反向更新，以及训练模型权重与部署ACTRefPolicy逐元素兼容；**不是黄块真实任务训练，也不会交付那个随机小模型**。

## 训练机检查

指定服务器是`zhaoruiqi_yizu@101.132.150.51:14802`。该服务器只接受公钥。旧项目连接脚本使用本机默认`~/.ssh/id_ed25519`；当前执行环境读取该文件失败，尚未认证成功。

登录后可用既有兼容Python运行`inventory_server.py`，只报告转换数据位置和GPU占用，不启动训练；再用`audit.py DATA_DIR`检查真实数据。必须是`yam_train/jpeg-v1`，不能直接给旧LeRobot的320×240数组。

## 执行方式

在服务器独立目录保存本仓库。使用现有且已验证的训练Python，或另建实验环境；不要升级其他人的共享环境。以下`DATA_DIR`与`NEW_OUTPUT_DIR`必须替换为检查确认的实际绝对路径：

```bash
# 先只做真实数据检查，不分配训练任务：
python training/train_one.py --data DATA_DIR --out NEW_OUTPUT_DIR

# 确认输出与空闲GPU后，显式启动这一个新实验：
python training/train_one.py --data DATA_DIR --out NEW_OUTPUT_DIR --launch
```

启动器先运行同一真实数据的2epoch烟雾训练并核对checkpoint合同，再从相同新初始化进入完整实验。输出`evaluation/yellow_red_seed42_selected.pt`后仍标记`robot_tested=false`；用现有部署入口单独验证、人工替换候选，不覆盖已跑通模型。

评估是固定验证窗口的开环误差，不是完整闭环成功率；要直接比较fufu旧权重，还必须确认旧模型未见过本次验证episode。若旧模型使用全部49段训练，就不能在其中10段上声称它是“未见测试集”。需要新采集的独立测试episode或单独说明该比较的训练重叠。

新的任务训练权重可以记录为你组织并完成的实验产物；ACT网络、训练器、ImageNet预训练和已有数据来源仍需准确注明，不因为换一个随机种子就将第三方架构声明为原创。

## 训练后与fufu比较并保留候选

`after_training.py`在本实验训练、选择结束后自动启动`compare_baseline.py`。它不改正在运行的训练器，也不改变当前机器人部署：

- 使用相同10段完整验证episode，对新模型和fufu留出验证版比较，检查二者训练episode均不包含验证episode。
- 另外比较fufu全部49段训练版，并明确标记它看过这些episode，不能称为泛化分数。
- 统一使用记录图像与qpos、30Hz、60步chunk、0.1秒blend、100Hz过滤和130ms一阶跟踪模拟；给出关节、夹爪误差、越界率、过滤次数与批量均摊GPU耗时。
- 模拟使用固定记录观测，模型动作不会改变下一帧图像，所以只能作离线目标／跟踪比较，不能代替物理抓取成功率。fufu训练版与实际部署导出文件是否逐字节一致也需要另行核对。
- 新模型仅在公平验证比较胜出，且相对fufu全数据基线不增加右臂／夹爪模拟误差和夹爪越界频率时推荐；否则保留fufu全数据基线作为离线推荐，不因“新训练”而强制换模型。
- `model_comparison/retained/`保留新模型和fufu全数据模型原件，`offline_recommended.pt`保存数值推荐，`comparison.json`给出依据。不会删除失败候选、不会覆盖现有上机权重。
