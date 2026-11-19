"""Factory integration tests; require the normal OpenPI training dependencies."""

import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np


_TRAINING_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("jax", "flax", "tyro", "torch", "etils", "lerobot")
)


@unittest.skipUnless(_TRAINING_AVAILABLE, "OpenPI training dependencies are not installed")
class G1DDataConfigTest(unittest.TestCase):
    def test_factories_repack_preserve_deltas_and_use_raw_action_key(self) -> None:
        from openpi import transforms
        from openpi.models import pi0_config
        from openpi.training import config

        for cls, dim in ((config.LeRobotG1DJointDataConfig, 16), (config.LeRobotG1DEEFDataConfig, 14)):
            with self.subTest(factory=cls.__name__), tempfile.TemporaryDirectory() as directory:
                factory = cls(repo_id="test/g1d", action_key="control.delta")
                with (
                    mock.patch.object(config.DataConfigFactory, "_load_norm_stats", return_value=None),
                    mock.patch.object(config.ModelTransformFactory, "__call__", return_value=transforms.Group()),
                ):
                    data_config = factory.create(pathlib.Path(directory), pi0_config.Pi0Config(pi05=True))
                self.assertEqual(data_config.action_sequence_keys, ("control.delta",) if dim == 16 else ("control.delta", "observation.state"))
                self.assertTrue(data_config.prompt_from_task)
                self.assertTrue(data_config.use_quantile_norm)
                actions = np.full((3, dim), 0.02, dtype=np.float32)
                sample = {
                    "observation.state": np.zeros(dim if dim == 16 else (3, dim), dtype=np.float32),
                    "control.delta": actions.copy(),
                    "prompt": "task",
                    **{
                        f"observation.images.{camera}": np.zeros((3, 12, 16), dtype=np.float32)
                        for camera in ("cam_left_high", "left_wrist", "right_wrist")
                    },
                }
                model_input = transforms.compose(
                    [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
                )(sample)
                np.testing.assert_allclose(model_input["actions"], actions, atol=1e-7)
                output = transforms.compose(data_config.data_transforms.outputs)(
                    {"state": model_input["state"], "actions": np.pad(actions, ((0, 0), (0, 32 - dim)))}
                )
                np.testing.assert_allclose(output["actions"], actions, atol=1e-7)

    def test_fixed_prompt_and_model_dimension_check(self) -> None:
        from openpi import transforms
        from openpi.models import pi0_config
        from openpi.training import config

        factory = config.LeRobotG1DEEFDataConfig(
            repo_id="test/g1d", state_dim=16, eef_state_key="observation.eef_state",
            prompt_from_task=False, default_prompt="fixed task"
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(config.DataConfigFactory, "_load_norm_stats", return_value=None),
            mock.patch.object(config.ModelTransformFactory, "__call__", return_value=transforms.Group()),
        ):
            data_config = factory.create(pathlib.Path(directory), pi0_config.Pi0Config())
            self.assertNotIn("prompt", data_config.repack_transforms.inputs[0].structure)
            self.assertFalse(data_config.prompt_from_task)
            with self.assertRaises(ValueError):
                factory.create(pathlib.Path(directory), pi0_config.Pi0Config(action_dim=8))


if __name__ == "__main__":
    unittest.main()
