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

启动命令加上 `--rtc` 可启用实时片段衔接（Real-Time Chunking）。客户端可以读取 `metadata["g1d"]["rtc"]` 确认服务能力。首次请求照常发送观测。后续重规划时，在采集观测的同一控制步截取旧计划尚未执行的部分，随观测一起发送：

```python
# Joint 示例：旧计划还剩 10 行，预计推理往返耗时 7 个控制步
observation["rtc_prefix"] = old_actions[next_index:next_index + 10]
observation["rtc_delay"] = 7
result = client.infer(observation)  # 在独立于控制循环的推理线程调用
```

`rtc_prefix` 形状为 `[K,D]`，要求 `0 <= rtc_delay <= K <= H` 且 `K > 0`。Joint 直接传绝对目标。EEF 客户端需先把旧目标换算到**本次观测的末端位姿**，沿用配置的 `eef_delta_frame` 和绝对夹爪值。模型采样时对延迟覆盖的行使用全权重，对剩余重叠行逐步减弱引导。等待回复时控制循环继续执行旧计划。若回复到达时已过去 `n` 个控制步，从新片段的第 `n` 行（从 0 开始）接续；若 `n >= H`，丢弃过期片段并重新请求。

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

## 4090 与宇树 G1-D 双端实测

### 1. 在 4090 上启动检查点服务

登录服务器后运行以下命令。测试和执行共用同一个服务，均保留 π0.5 完整的 **10 步采样**。该检查点每轮返回 25 帧绝对关节和夹爪目标。

XLA_PYTHON_CLIENT_PREALLOCATE=false
```bash
cd ~/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run -m examples.g1d.main \
  --config pi05_g1d \
  --checkpoint "$PWD/checkpoints/pi05_g1d/shelf_pp_test/25000" \
  --host 192.168.66.200 --port 8000
```

### 2. 在机器人上测试通信和预热延迟

从另一终端登录机器人：

```bash
ssh unitree@192.168.66.47
```

在机器人终端运行以下命令。测试默认只打印预测。第一次请求用于预热，随后 20 次的 `inference round trip` 可用于统计中位数和 P95。

Pick the green bottle from the shelf and place it at the fixed target.
Pick the blue bottle from the shelf and place it at the fixed target.

source ~/miniconda3/etc/profile.d/conda.sh
```bash
conda activate tv
cd ~/zhoumy/xr_teleoperate
POLICY_URL=ws://192.168.66.200:8000
PROMPT="Pick the green bottle from the shelf and place it at the fixed target."

python unitree_openpi/scripts/test_g1d_client.py \
  --url "$POLICY_URL" --prompt "$PROMPT" --max-steps 20
```

可选任务文本：上面的英文是已用于链路实测的示例。可将 `PROMPT` 替换为训练数据中对应任务的原文。仅检查通信时也可用 `PROMPT="test"`；测试脚本省略 `--prompt` 时默认使用 `test`。

### 3. 在机器人上执行一次策略响应

**先核对夹爪单位和端点。** 此检查点此前输出过约 2.6 和 5.4 的夹爪目标，超过机器人当前配置的 0–2.3。客户端会拒绝下发越界目标。确认模型输出与真实设备量程匹配后，在上一步的机器人终端运行：

```bash
python unitree_openpi/scripts/run_g1d_client.py \
  --url "$POLICY_URL" --prompt "$PROMPT" --max-steps 1 --execute
```

`--max-steps 1` 限制为一轮推理；机器人配置中的 `execute_steps` 决定这一轮采用多少帧，当前为 1。去掉 `--execute` 可用相同入口只打印目标。

### 4. 保留 10 步采样时的通信优化

4090 与机器人 Wi-Fi 的一次实测中，重复发送同一份真实观测，预热后 20 轮往返中位数为 **127.8 ms**，其中服务端策略调用为 **103.9 ms**，外围耗时约 **24 ms**。机器人打包约 0.32 ms、解包约 0.12 ms。三张 RGB 图像的原始数据共 451,584 字节，动作回复约 1.7 KB。这组数据不包含等待相机新帧的时间。

优先比较 Wi-Fi 和千兆有线：保持同一检查点、三路图像、prompt 和采样步数，分别记录预热后的往返和服务端计时。在 Wi-Fi 上还可对比两端配套的无损图像压缩，把编码、传输和解码全部计入往返，并核对解码后的像素。以当前基线，即使消除全部通信与解包耗时，服务端策略调用本身仍超过 80 ms。

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

覆盖云端 Joint 固定参考还原、Joint/EEF输入输出、训练片段重定位、默认prompt、元信息和原客户端多轮WebSocket收发。完整工厂测试需要训练依赖，缺少时显式跳过。双端实测步骤见上文。
