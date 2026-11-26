"""Cloud protocol checks; no camera or robot SDK is involved."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from openpi_client import base_policy

from examples.g1d import real_env
import openpi.policies
from openpi.policies import g1d_policy
import openpi.training


def observation(dim: int = 16) -> dict:
    return {
        "observation/state": np.full(dim, 10, dtype=np.float32),
        **{key: np.zeros((12, 16, 3), dtype=np.uint8) for key in g1d_policy.CAMERA_KEYS.values()},
        "prompt": "task",
    }


class RecordingPolicy(base_policy.BasePolicy):
    def __init__(self, dim: int = 16, horizon: int = 2) -> None:
        self.dim, self.horizon = dim, horizon
        self.last_obs = None

    def infer(self, obs: dict) -> dict:
        self.last_obs = obs
        return {"actions": np.full((self.horizon, self.dim), 0.1, dtype=np.float32)}


def environment(policy, *, action_space="joint", state_dim=None, action_horizon=2, **kwargs):
    """Explicit protocol fixture, independent of the production CLI."""
    action_dim = 16 if action_space == "joint" else 14
    return real_env.G1DRealEnvironment(
        policy,
        action_space=action_space,
        state_dim=action_dim if state_dim is None else state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        **kwargs,
    )


class G1DRealEnvironmentTest(unittest.TestCase):
    def test_rtc_references_and_weights(self) -> None:
        policy = mock.Mock()
        policy.infer.return_value = {"actions": np.zeros((4, 16))}
        obs = observation(17)
        obs["observation/state"][0] = 1
        obs.update(rtc_prefix=np.full((3, 16), 11), rtc_delay=1)
        for absolute in (False, True):
            env = environment(policy, rtc=True, state_dim=17, action_horizon=4, absolute_joint_targets=absolute)
            env.infer(obs)
            args, kwargs = policy.infer.call_args
            expected = np.full((3, 16), 11 if absolute else 1)
            expected[:, [7, 15]] = 11
            np.testing.assert_array_equal(args[0]["actions"][:3], expected)
            np.testing.assert_allclose(kwargs["rtc_weights"][:, 0], [1, 2/3, 1/3, 0])

    def test_rtc_delay_must_be_covered_by_prefix(self) -> None:
        obs = observation()
        obs.update(rtc_prefix=np.ones((1, 16)), rtc_delay=2)
        policy = mock.Mock()
        with self.assertRaisesRegex(ValueError, "rtc_delay"):
            environment(policy, rtc=True).infer(obs)
        policy.infer.assert_not_called()

    def test_absolute_joint_and_relative_eef_responses(self) -> None:
        for space, dim in (("joint", 16), ("eef", 14)):
            with self.subTest(space=space):
                policy = RecordingPolicy(dim)
                env = environment(policy, action_space=space, action_horizon=2)
                obs = observation(dim)
                obs["telemetry"] = {"temperature": 30}
                result = env.infer(obs)
                expected = np.full((2, dim), 0.1, dtype=np.float32)
                if space == "joint":
                    expected[:, :7] += 10
                    expected[:, 8:15] += 10
                np.testing.assert_allclose(result["actions"], expected)
                np.testing.assert_allclose(policy.last_obs["observation/state"], 10)
                self.assertNotIn("telemetry", policy.last_obs)
                self.assertIn("telemetry", obs)

    def test_default_prompt(self) -> None:
        policy = RecordingPolicy()
        env = environment(policy, action_horizon=2, default_prompt="default task")
        obs = observation()
        del obs["prompt"]
        env.infer(obs)
        self.assertEqual(policy.last_obs["prompt"], "default task")
        self.assertNotIn("prompt", obs)

    def test_invalid_wire_inputs_are_rejected_before_model(self) -> None:
        for key, value in (
            ("observation/state", np.zeros(14)),
            ("observation/state", np.full(16, np.inf)),
            ("observation/images/high", np.zeros((3, 12, 16), dtype=np.uint8)),
            ("observation/images/high", np.zeros((12, 16, 3), dtype=np.float32)),
            ("prompt", ""),
        ):
            with self.subTest(key=key):
                policy = RecordingPolicy()
                env = environment(policy, action_horizon=2)
                obs = observation()
                obs[key] = value
                with self.assertRaises(ValueError):
                    env.infer(obs)
                self.assertIsNone(policy.last_obs)
        obs = observation()
        del obs["observation/images/right_wrist"]
        with self.assertRaises(KeyError):
            env.infer(obs)

    def test_invalid_output_shape_and_numbers_are_rejected(self) -> None:
        for policy in (RecordingPolicy(14), RecordingPolicy(horizon=3)):
            with self.subTest(dim=policy.dim, horizon=policy.horizon), self.assertRaises(ValueError):
                environment(policy, action_horizon=2).infer(observation())
        policy = RecordingPolicy()
        policy.infer = lambda obs: {"actions": np.full((2, 16), np.nan)}
        with self.assertRaises(ValueError):
            environment(policy, action_horizon=2).infer(observation())

    def test_eef_joint_feedback_needs_no_execution_pose_on_cloud(self) -> None:
        env = environment(
            RecordingPolicy(14), action_space="eef", state_dim=16, action_horizon=2
        )
        self.assertEqual(env.infer(observation())["actions"].shape, (2, 14))
        contract = env.metadata["g1d"]
        self.assertEqual(contract["state_representation"], "absolute")
        self.assertEqual(contract["action_reference"], "request_observation_state")
        self.assertEqual(contract["gripper_representation"], "absolute")


class EnvironmentLoadingTest(unittest.TestCase):
    """Verify config propagation without importing a GPU stack or weights."""

    def loading_context(self, data_factory):
        class JointFactory:
            pass
        class EefFactory:
            action_space = "eef"
            state_dim = 16
            action_dim = 14
            eef_delta_frame = "tool"
            default_prompt = "factory task"
        train_config = SimpleNamespace(
            data=EefFactory() if data_factory else object(),
            model=SimpleNamespace(action_horizon=7),
        )
        policy = RecordingPolicy(dim=14, horizon=7)
        policy.metadata = {"model": "fixture"}
        config_module = SimpleNamespace(
            get_config=mock.Mock(return_value=train_config),
            LeRobotG1DJointDataConfig=JointFactory,
            LeRobotG1DEEFDataConfig=EefFactory,
        )
        policy_module = SimpleNamespace(create_trained_policy=mock.Mock(return_value=policy))
        return config_module, policy_module, train_config, policy

    def test_contract_is_resolved_entirely_from_training_config(self) -> None:
        config_module, policy_module, train_config, policy = self.loading_context(True)
        with (
            mock.patch.object(openpi.training, "config", config_module, create=True),
            mock.patch.object(openpi.policies, "policy_config", policy_module, create=True),
        ):
            env = real_env.create_environment("eef_config", "/checkpoint", pytorch_device="cuda:0")
        contract = env.metadata["g1d"]
        self.assertEqual(contract["state_dim"], 16)
        self.assertEqual(contract["action_dim"], 14)
        self.assertEqual(contract["action_horizon"], 7)
        self.assertEqual(contract["action_space"], "eef")
        self.assertEqual(contract["eef_delta_frame"], "tool")
        self.assertEqual(env.metadata["model"], "fixture")
        policy_module.create_trained_policy.assert_called_once_with(
            train_config, "/checkpoint", default_prompt="factory task", pytorch_device="cuda:0"
        )
        obs = observation()
        del obs["prompt"]
        self.assertEqual(env.infer(obs)["actions"].shape, (7, 14))
        self.assertEqual(policy.last_obs["prompt"], "factory task")

    def test_wrong_data_factory_is_rejected_before_loading_weights(self) -> None:
        config_module, policy_module, _, _ = self.loading_context(False)
        with (
            mock.patch.object(openpi.training, "config", config_module, create=True),
            mock.patch.object(openpi.policies, "policy_config", policy_module, create=True),
        ):
            with self.assertRaises(ValueError):
                real_env.create_environment("wrong_config", "/checkpoint")
        policy_module.create_trained_policy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
