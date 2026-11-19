# G1-D 云端 WebSocket 推理

OpenPI 只负责接收机器人通过 WebSocket 上传的绝对状态、RGB 图像和任务文本，运行模型并返回动作片段。相机驱动、状态采集、时间同步、IK 和动作执行均由机器人侧系统负责。

| 文件 | 云端职责 |
| --- | --- |
| [main.py](main.py) | 解析推理启动参数、创建环境并启动项目原有 WebSocket server |
| [real_env.py](real_env.py) | 从训练配置加载模型、解析网络数据契约，校验请求和响应 |
| [training.md](training.md) | Joint/EEF 数据工厂、动作参考及训练转换说明 |
| [test_main.py](test_main.py)、[test_real_env.py](test_real_env.py) | 协议、入口和原客户端互操作测试 |

`real_env` 是远端机器人的网络适配层，按现有 server 接口实现 `BasePolicy.infer()`；不继承本地控制 Runtime 的 Environment，也不连接硬件。本目录不另写 WebSocket 协议，直接复用 [WebsocketPolicyServer](../../src/openpi/serving/websocket_policy_server.py) 和 [WebsocketClientPolicy](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py)。

## 数据边界

```text
机器人系统：采集状态/图像，组织观测
   ── WebSocket / MessagePack + NumPy ──→
OpenPI：real_env 校验 → G1DInputs → 归一化 → 模型 → 反归一化 → G1DOutputs → Joint 加回本轮关节角
   ←── WebSocket / actions[H,D] ────────
机器人系统：接收绝对关节目标、动作缓冲、执行与反馈
```

请求示例：

```python
observation = {
    "observation/state": state,                  # float32 [S]，绝对反馈
    "observation/images/high": head_rgb,       # uint8 HWC RGB
    "observation/images/left_wrist": left_rgb,   # uint8 HWC RGB
    "observation/images/right_wrist": right_rgb, # uint8 HWC RGB
    "prompt": "pick up the object",
}
```

OpenPI 不关注这些数据如何从设备读取，只检查约定字段、维度、dtype和有效数值。图像缩放/分词/归一化是模型预处理，与底层采集分离，仍由标准模型流程处理。

响应为 `{"actions": array[H,D], "policy_timing": ..., "server_timing": ...}`。

- Joint 默认 state/action 各16维：左臂7关节+夹爪、右臂7关节+夹爪。
- EEF 默认 state/action 各14维：每臂xyz(3)+rotvec(3)+夹爪；也支持16维关节反馈作为模型state，动作仍14维。
- Joint 云端逐步将模型 delta 加回同一个请求状态 s0，返回绝对关节目标和绝对夹爪目标。EEF 返回相对位姿和绝对夹爪目标。
- EEF旋转/位置参考按训练配置的 `eef_delta_frame`；使用关节反馈作为模型state时，机器人侧根据同一时刻的实际EEF位姿还原目标，不必额外向云端上传执行用的EEF参考。

连接后先发送 metadata，其中 `g1d` 公布维度、相机字段、`action_representation=absolute（Joint）或 relative（EEF）`、`gripper_representation=absolute`、EEF坐标系等。其后一次请求对应一个动作片段响应，不包含执行确认或主动状态订阅。输入或推理错误沿用原 server 的文本错误帧和断开行为。

## 启动

在云端已经安装项目依赖的环境，从仓库根目录运行：

```bash
uv run python -m examples.g1d.main --config pi05_g1d --checkpoint YOUR_CHECKPOINT --port 8000
```

所选配置必须使用 `LeRobotG1DJointDataConfig` 或 `LeRobotG1DEEFDataConfig`，并已用真实数据训练得到匹配权重。当前 `pi05_g1d` 使用 Joint 工厂和本地数据/资产路径；需核对实际数据定义，并计算统计和训练得到匹配权重。

`main` 只接受推理服务启动参数：`--config`、`--checkpoint`、`--host`、`--port`、`--default-prompt`、`--pytorch-device`。不提供训练参数的命令行覆盖，也不在入口内重算或设置维度。

`real_env.create_environment()` 根据配置名加载训练配置和策略，并直接从配置构造网络契约：

| 数据约定 | 唯一来源 |
| --- | --- |
| 动作空间 | `train_config.data.action_space` |
| state/action维度 | `train_config.data.state_dim/action_dim` |
| 预测长度 | `train_config.model.action_horizon` |
| EEF坐标系 | `train_config.data.eef_delta_frame` |

因此切换 Joint/EEF 只需换训练配置和匹配的 checkpoint。`--default-prompt` 用于推理时的默认任务文本；未指定时沿用数据工厂的默认文本。`--pytorch-device cuda:0` 指定推理设备。

协议测试中的模拟策略只放在测试代码中，不通过生产入口配置维度或选择 mock。真实服务启动需要完整模型环境、训练配置和权重。

轻量协议单元测试使用 Python 3.11+、`openpi-client` 和兼容的 websockets，并使本仓库 `src` 可导入。PowerShell 示例：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\packages\openpi-client\src;$PWD"
```

## 机器人侧使用原客户端

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="SERVER_IP", port=8000)
metadata = client.get_server_metadata()
result = client.infer(observation)
action_chunk = result["actions"]
# Joint 的 actions 已是绝对目标，机器人侧直接使用。
```

现有客户端为同步请求响应；运行中超时重连、调度和断线处理沿用项目当前能力，本目录不实现机器人底层控制。`GET /healthz` 是原 server 的存活检查，不代表机器人状态或模型已预热。

## 测试

```bash
python -m unittest openpi.policies.g1d_policy_test openpi.training.g1d_config_test examples.g1d.test_real_env examples.g1d.test_main -v
```

覆盖云端 Joint 固定参考还原、Joint/EEF输入输出、训练片段重定位、默认prompt、元信息和原客户端多轮WebSocket收发。完整工厂测试需要训练依赖，缺少时显式跳过；尚未运行真实G1-D模型或实机。
