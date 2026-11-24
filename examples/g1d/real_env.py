"""G1-D cloud-side environment: network observations in, joint targets or EEF deltas out.

This adapter has no camera, robot SDK, state collector, IK or actuator interface.
WebsocketPolicyServer calls infer() with the observation supplied by the robot.
"""

from typing import Literal

import numpy as np
from openpi_client import base_policy

from openpi.policies import g1d_policy


class G1DRealEnvironment(base_policy.BasePolicy):
    """Validate one remote observation and its matching prediction response."""

    def __init__(
        self,
        policy: base_policy.BasePolicy,
        *,
        action_space: Literal["joint", "eef"],
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        default_prompt: str | None = None,
        eef_delta_frame: Literal["base", "tool"] = "base",
        metadata: dict | None = None,
    ) -> None:
        if action_space not in ("joint", "eef") or eef_delta_frame not in ("base", "tool"):
            raise ValueError("Choose joint/eef action space and base/tool EEF delta frame")
        if min(state_dim, action_dim, action_horizon) <= 0:
            raise ValueError("State/action dimensions and action horizon must be positive")
        if action_space == "joint" and (state_dim, action_dim) not in ((16, 16), (17, 16)):
            raise ValueError("Joint mode requires 16-D absolute joint and gripper vectors")
        self._action_space = action_space
        self.action_dim = action_dim
        self.state_dim = state_dim
        self._policy = policy
        self._horizon = action_horizon
        self._default_prompt = default_prompt
        self.metadata = {
            **(metadata or {}),
            "g1d": {
                "protocol": "openpi-observation-v1",
                "state_key": "observation/state",
                "state_dim": self.state_dim,
                "state_representation": "absolute",
                "image_keys": list(g1d_policy.CAMERA_KEYS.values()),
                "image_format": "RGB uint8 HWC",
                "action_space": action_space,
                "action_dim": self.action_dim,
                "action_horizon": action_horizon,
                "action_representation": "absolute" if action_space == "joint" else "relative",
                "action_reference": None if action_space == "joint" else "request_observation_state",
                "gripper_representation": "absolute",
                **({"gripper_units": "motor_position"} if action_space == "joint" else {}),
                "eef_delta_frame": eef_delta_frame if action_space == "eef" else None,
            },
        }

    def infer(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            raise ValueError("Observation must be a dictionary")
        inputs = {"observation/state": g1d_policy.validate_state(obs["observation/state"], self.state_dim)}
        if self._action_space == "joint" and self.state_dim == 17 and inputs["observation/state"][0] not in (0, 1):
            raise ValueError("state[0] must be a binary scan_success flag")
        for key in g1d_policy.CAMERA_KEYS.values():
            inputs[key] = g1d_policy.parse_rgb_image(obs[key], wire_format=True)
        prompt = obs.get("prompt", self._default_prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Send a nonempty prompt or configure --default-prompt")
        inputs["prompt"] = prompt
        result = dict(self._policy.infer(inputs))
        result["actions"] = g1d_policy.validate_actions(
            result["actions"], self.action_dim, horizon=self._horizon
        )
        if self._action_space == "joint":
            reference = inputs["observation/state"][-16:]
            result["actions"] = np.stack([
                g1d_policy.absolute_target(row, reference, "joint") for row in result["actions"]
            ])
        return result

    def reset(self) -> None:
        self._policy.reset()


def create_environment(
    config_name: str,
    checkpoint: str,
    *,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
) -> G1DRealEnvironment:
    """Load policy and resolve the network contract from its training config.

    The caller chooses a checkpoint, not dimensions or action representations.
    Deferred imports keep protocol unit tests independent of GPU dependencies.
    """
    from openpi.policies import policy_config
    from openpi.training import config

    train_config = config.get_config(config_name)
    data_factory = train_config.data
    if not isinstance(data_factory, (config.LeRobotG1DJointDataConfig, config.LeRobotG1DEEFDataConfig)):
        raise ValueError("The training config must use LeRobotG1DJointDataConfig or LeRobotG1DEEFDataConfig")
    prompt = default_prompt if default_prompt is not None else data_factory.default_prompt
    policy = policy_config.create_trained_policy(
        train_config, checkpoint, default_prompt=prompt, pytorch_device=pytorch_device
    )
    return wrap_policy(policy, train_config, config_name, prompt)


def wrap_policy(policy, train_config, config_name: str, prompt: str | None) -> G1DRealEnvironment:
    data_factory = train_config.data
    return G1DRealEnvironment(
        policy,
        action_space=data_factory.action_space,
        state_dim=data_factory.state_dim,
        action_dim=data_factory.action_dim,
        action_horizon=train_config.model.action_horizon,
        default_prompt=prompt,
        eef_delta_frame=data_factory.eef_delta_frame,
        metadata={**policy.metadata, "config": config_name},
    )
