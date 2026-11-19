# G1-D policy 与 LeRobot 数据配置

## 两者分别负责什么

| 模块 | 职责 | 不在此层处理 |
| --- | --- | --- |
| [g1d_policy.py](../../src/openpi/policies/g1d_policy.py) | 对一个样本执行 `G1DInputs` / `G1DOutputs`：状态检查、训练片段参考重定位、图像和模型字段映射、输出去除 padding | 不打开数据集，不执行模型，不做实际机器人控制 |
| [LeRobotG1DJointDataConfig / LeRobotG1DEEFDataConfig](../../src/openpi/training/config.py) | 指定数据集字段/动作维度，创建 repack、G1-D转换、模型转换，加载对应统计，组成完整 `DataConfig` | 不替代转换器逐样本处理，不执行硬件 |
| 标准 `Policy` / 数据加载器 | 按 DataConfig 装配的顺序运行转换、归一化及模型处理 | 不替你选择实际 SDK |
| [real_env.py](real_env.py) | 云端网络适配：校验 WebSocket 观测、调用策略、检查返回动作和公开数据契约 | 不采集相机/状态，不调用 SDK，不执行目标 |

工厂是“如何组装整条数据流水线的配方”，`g1d_policy` 是“流水线里逐样本运行的机器人转换步骤”。工厂会创建和使用 `G1DInputs/G1DOutputs`，不是两套不同的推理模型。

云端启动入口不重复设置这些训练参数。`main` 只选择配置/权重和推理服务参数，`real_env.create_environment()` 从训练配置读取state/action维度、动作空间、预测长度和EEF坐标系，并公开到网络metadata。

## 统一的相对动作定义

**state 为当前绝对反馈；片段内每个机械臂 action 都相对这次输入 state，夹爪为绝对值。不是相邻控制步之间的增量。**

云端回包前还原 Joint：`q_target[t+k] = q_observation[t] + action[k]`，仅机械臂通道相加。EEF 位置和旋转按所选坐标系相对同一参考位姿还原。Joint 回包为绝对目标；EEF 回包保留相对位姿。

| 工厂 | 默认 state | action | 夹爪索引 |
| --- | --- | --- | --- |
| `LeRobotG1DJointDataConfig` | 16维：左q(7)、夹爪、右q(7)、夹爪 | 16维：左dq(7)、夹爪、右dq(7)、夹爪 | 7、15 |
| `LeRobotG1DEEFDataConfig` | 14维：左xyz(3)、rotvec(3)、夹爪、右同样 | 14维：左dxyz(3)、drotvec(3)、夹爪、右同样 | 6、13 |

EEF 可设 `state_dim=16`，用关节反馈作为模型输入。机器人侧此时需掌握同一时刻的绝对EEF位姿作为执行参考，但无需为在线模型推理额外上传。EEF旋转增量默认使用旋转向量（rad），默认基坐标系；执行端支持显式选择工具坐标系。Joint 模式的夹爪使用绝对电机位置，归一化和反归一化由云端策略处理。

Joint 默认读取绝对目标，每个目标减去片段首帧 s0 的关节角，夹爪保留绝对值。pi05_g1d 使用 MarketDemo 的夹爪末置布局，离线时重排；线上始终使用左右臂各自紧跟夹爪的布局。旧的逐行相对 Joint 数据可配置 absolute_joint_targets=False。以下重定位说明适用于逐行相对动作。训练时会将各行的局部 state 参考重定位到片段首帧；Joint 推理在 real_env 加回本轮 s0 后回包。模型输入补齐到32维；反归一化后由 `G1DOutputs` 返回16/14维。

## 数据集片段的参考必须正确

绝对 Joint 只取动作序列和单帧 state。逐行相对模式的 `action_sequence_keys` 同时包含 `action_key` 和 `state_key`，让 LeRobot 读取器读取 H 个时间点的动作和绝对状态。`G1DInputs` 将状态序列的第一帧作为模型 state，并将每行相对动作统一到该帧参考。在线推理收到单个 state 向量时则不进行训练序列重定位。

因此，如果原始第 `t+k` 行存的是 `target[t+k] - state[t+k]`，把这些行直接堆叠后并不等于相对 `state[t]` 的整个片段。Joint 情况应先还原每行目标，再重定位：

```text
stored_delta[k] = target[t+k] - state[t+k]
chunk_delta[k]  = state[t+k] + stored_delta[k] - state[t]
```

夹爪不参与上述加减。EEF 旋转重定位需要正确的姿态组合，不能直接对欧拉角或旋转向量做同样减法。

当前实现已完成 Joint 和 EEF 的上述训练重定位，EEF 位移及旋转按 `eef_delta_frame="base"/"tool"` 组合；夹爪始终原样保留。原始 LeRobot 数据默认约定是“每行 action 相对该行绝对 state”。如果自定义读取器已经提供相对片段首帧的完整 chunk，应传单个参考 state 向量和该 chunk，不再提供逐行参考序列，避免重复重定位。

EEF 若使用 `state_dim=16` 的关节反馈，设置 `eef_state_key="observation.eef_state"`（替换为真实字段），工厂会额外读取 H 帧绝对 EEF 位姿作为训练参考，并保留关节首帧作为模型输入。该训练字段每行14维，布局与默认 EEF state 相同；不能用关节角代替末端姿态。在线云端只处理单帧模型state，不进行训练片段重定位，因此不要求额外上传该训练参考字段。

## 数据工厂的字段映射

| 可配置参数 | 默认 LeRobot 字段 | G1-D转换接受的字段 |
| --- | --- | --- |
| `state_key` | `observation.state` | `observation/state` |
| `action_key` | `action` | `actions` |
| `head_image_key` | `observation.images.cam_left_high` | `observation/images/high` |
| `left_wrist_image_key` | `observation.images.left_wrist` | `observation/images/left_wrist` |
| `right_wrist_image_key` | `observation.images.right_wrist` | `observation/images/right_wrist` |

图像均为有效相机，映射到模型 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。转换兼容训练 float CHW 和在线 uint8 HWC RGB。

默认 `prompt_from_task=True`，从 LeRobot task 提取 prompt；固定任务可设置 `prompt_from_task=False, default_prompt="..."`。训练 repack 不自动用于普通在线推理；环境负责发送右侧规范字段。

## 训练配置示例

当前 `pi05_g1d` 使用 EEF 数据工厂，并按给定的本地路径配置数据和归一化资产。数据实际字段、维度和动作定义仍需匹配；下面是两种工厂的配置项示例：

```python
# Joint，放入 TrainConfig(...)
model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
data=LeRobotG1DJointDataConfig(repo_id="YOUR_NAMESPACE/YOUR_JOINT_DATASET"),
```

```python
# EEF，放入另一个 TrainConfig(...)
model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
data=LeRobotG1DEEFDataConfig(repo_id="YOUR_NAMESPACE/YOUR_EEF_DATASET"),
```

Joint/EEF使用独立配置名与统计资源，在正确重定位后的训练样本上分别计算归一化统计。沿用同一个 `repo_id` 而不区分资源可能误用统计。模型内部32维与真实动作16/14维不同；动作采样频率取决于数据集fps，修改H不会自动重采样。

## 本地数据与绝对资产目录

```python
repo_id="/home/gming/projects/openpi/datasets/bipiper-box-shortv2",
assets=AssetsConfig(
    assets_dir="/home/gming/projects/openpi/datasets",
    asset_id="bipiper-box-shortv2",
),
```

将上述配置项放在数据工厂中。数据加载器将以 `/` 开头的 `repo_id` 作为本地root；统计脚本保存到 `assets_dir / asset_id / norm_stats.json`，数据工厂也从相同目录读取。未指定 `assets_dir` 时仍使用训练配置的默认assets目录。

这里统计文件为 `/home/gming/projects/openpi/datasets/bipiper-box-shortv2/norm_stats.json`。`asset_id` 保持相对名称，checkpoint仍将统计复制到自身的 `assets/bipiper-box-shortv2/norm_stats.json`，推理从checkpoint读取，不依赖原数据目录。

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_g1d
```

## 参考了哪个模板

LeRobot 是存储/读取格式；ALOHA、LIBERO、G1-D 是机器人或环境语义。本实现是 LeRobot存储 + G1-D专用转换。

工厂结构参考 `LeRobotLiberoDataConfig.create()` 的 repack/data/model transforms 组装；`G1DInputs/Outputs` 参考 LIBERO的字典转换与输出裁剪，三路相机组织参考 ALOHA。没有继承 LIBERO的7维输出，也没有沿用 ALOHA的关节符号、14维关节布局或夹爪标定。

ALOHA和部分LIBERO配置中的 `DeltaActions` 通常将绝对动作转成相对片段起始state的值。这里数据已相对，不重复转换；Joint 由云端通过固定 state 参考还原目标。

## DataConfigFactory.create 为什么用工厂

它是工厂方法模式的一种实现，工厂本身也是普通类实例：

```python
factory = LeRobotG1DJointDataConfig(repo_id="my_org/my_dataset")
resolved = factory.create(train_config.assets_dirs, train_config.model)
```

`factory` 保存创建配方；`resolved` 是装配好的 `DataConfig`。创建时才知道资产目录、模型类型、token参数、补齐维度等，因此在 `create()` 中统一加载统计和组装流水线，避免重复配置及导入时就提前初始化。

当然可以直接构造 `DataConfig(...)`；但现有框架统一调用 `config.data.create(...)`，直接传入结果类需要调整接口或增加适配。工厂和“用类”不是互斥关系，差别是把创建配方与最终创建结果分开。

## G1-D policy 的实现分层与 ALOHA 区别

G1-D 外部图像名称为 `high`，默认数据集字段 `observation.images.cam_left_high`，在线字段 `observation/images/high`，由 `head_image_key` 配置。映射到模型内部仍为 `base_0_rgb`，这是模型固定接口，不表示物理底座相机。

额外绝对 EEF 位姿字段的参数名为 `eef_state_key`，训练 repack 后为 `observation/eef_state`。仅当模型 state 不是默认14维EEF位姿时需要它；在线请求不要求为执行额外上传该字段。

`g1d_policy.py` 包含四类逻辑：

1. `validate_state` / `validate_actions`：检查shape、有限数值及float32表示，动作输出可去除模型padding；不做统计归一化。
2. `parse_rgb_image`：兼容训练float CHW与在线uint8 HWC。在线协议额外严格要求uint8 HWC；不做设备采集或颜色标定。
3. `absolute_target` / `rebase_actions` 与旋转辅助函数：训练时将每行相对动作重定位到片段首帧。Joint通过位置偏移处理，EEF通过位移/姿态组合处理。`absolute_target`只是数学中间步骤，不发送机器人命令。
4. `G1DInputs` / `G1DOutputs`：把上述逻辑组合成逐样本转换。输入组装state、三路image和image_mask、prompt及可选actions；输出裁剪成16/14维，保留相对动作与绝对夹爪。

ALOHA同样使用Inputs/Outputs接口，但其双臂是6+1+6+1布局，且可选 `adapt_to_pi` 执行关节符号和专属夹爪标定。G1-D Joint为7+1+7+1，EEF为6+1+6+1，但同为14维不代表其EEF动作能套用ALOHA关节语义。

ALOHA配套DataConfig默认用DeltaActions把绝对目标减去片段首帧state，输出再用AbsoluteActions加回，因此默认完整流水线返回绝对目标。G1-D数据已相对，所以进行参考重定位，在线输出仍保留相对值。这是数据/协议定义带来的差别，不是两个模型采用了不同推理算法。

