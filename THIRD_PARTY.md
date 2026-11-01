# 代码来源与贡献范围

## 上游应用

- 来源：<https://github.com/FuFutheBest/yam_infer>
- 固定提交：`bb2e8aba33d6bd9bcba43c2ffca129ad90c3feb5`
- 保留目录：`project/yam_infer`、`project/third_party`、`project/tests`、上游配置、脚本和文档。
- 核心ACT、调度、worker、SDK和归位行为目前保留；不声称第三方实现为本项目作者原创。
- 分发副本的`project/SEVERS.md`已移除其中的登录密码。

## 上游已有第三方组件

来源和局部改动见 [project/VENDORED.md](project/VENDORED.md)。

- I2RT SDK：`project/third_party/i2rt`，保留其MIT许可证。
- ACT/DETR代码：`project/yam_infer/policy/detr`，保留其许可证和文件头。
- worker/config来源：SuperInference指定版本，按上游清单记录。

上游顶层没有独立的许可文件；本项目不对其代码重新授权。公开再分发时应先确认所需许可，组件已有许可证继续有效。当前仓库按私有保存。

## 本次新增

任务选择、独立环境安装、模型登记、哈希放行、无硬件验证和本地启动工具，以及黄块／红容器任务文档和验证记录。后续对最初`task1_act_runtime`的重构会通过独立提交记录设计与实现，不隐藏实际复用的依赖。
