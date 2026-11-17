"""Tests for G1-D observation-relative action preservation; no GPU dependencies required."""

import unittest

import numpy as np

from openpi.policies import g1d_policy


class G1DPolicyTest(unittest.TestCase):
    def observation(self, dim: int) -> dict:
        return {
            "observation/state": np.arange(dim, dtype=np.float32) + 10,
            **{key: np.full((12, 16, 3), 128, dtype=np.uint8) for key in g1d_policy.CAMERA_KEYS.values()},
            "prompt": "pick up the object",
        }

    def test_relative_actions_and_absolute_grippers_are_unchanged(self) -> None:
        for dim, grippers in ((16, (7, 15)), (14, (6, 13))):
            with self.subTest(dim=dim):
                obs = self.observation(dim)
                actions = np.full((3, dim), 0.02, dtype=np.float32)
                actions[:, grippers] = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]]
                obs["actions"] = actions.copy()
                transformed = g1d_policy.G1DInputs(dim, dim)(obs)
                np.testing.assert_array_equal(transformed["actions"], actions)
                padded = np.pad(transformed["actions"], ((0, 0), (0, 32 - dim)))
                output = g1d_policy.G1DOutputs(dim)({"actions": padded, "state": obs["observation/state"]})
                np.testing.assert_array_equal(output["actions"], actions)
                np.testing.assert_array_equal(obs["actions"], actions)
                self.assertEqual(set(output), {"actions"})

    def test_joint_training_chunk_rebases_each_row_to_first_state(self) -> None:
        obs = self.observation(16)
        states = np.stack([np.full(16, value, dtype=np.float32) for value in (1, 2, 3)])
        actions = np.full((3, 16), 0.1, dtype=np.float32)
        actions[:, [7, 15]] = [0.25, 0.75]
        obs["observation/state"], obs["actions"] = states, actions
        inputs = g1d_policy.G1DInputs()(obs)
        np.testing.assert_array_equal(inputs["state"], states[0])
        np.testing.assert_allclose(inputs["actions"][:, 0], [0.1, 1.1, 2.1])
        np.testing.assert_allclose(inputs["actions"][:, [7, 15]], [[0.25, 0.75]] * 3)
        for index in range(3):
            expected = g1d_policy.absolute_target(actions[index], states[index], "joint")
            actual = g1d_policy.absolute_target(inputs["actions"][index], states[0], "joint")
            np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actions[:, 0], 0.1)  # Source data remains unchanged.

    def test_eef_training_rebase_preserves_absolute_targets_in_both_frames(self) -> None:
        references = np.zeros((2, 14), dtype=np.float32)
        references[0, :6] = [1, 2, 3, 0, 0, 0.6]
        references[1, :6] = [2, 3, 4, 0.4, 0.3, 0]
        actions = np.full((2, 14), 0.1, dtype=np.float32)
        actions[:, [6, 13]] = [0.25, 0.75]
        for frame in ("base", "tool"):
            with self.subTest(frame=frame):
                rebased = g1d_policy.rebase_actions(actions, references, "eef", eef_delta_frame=frame)
                for index in range(2):
                    expected = g1d_policy.absolute_target(
                        actions[index], references[index], "eef", eef_delta_frame=frame
                    )
                    actual = g1d_policy.absolute_target(
                        rebased[index], references[0], "eef", eef_delta_frame=frame
                    )
                    np.testing.assert_allclose(actual, expected, atol=1e-6)
                np.testing.assert_allclose(rebased[:, [6, 13]], actions[:, [6, 13]])

    def test_eef_training_with_joint_state_uses_separate_reference_sequence(self) -> None:
        obs = self.observation(16)
        obs["observation/state"] = np.ones((2, 16), dtype=np.float32)
        reference = np.zeros((2, 14), dtype=np.float32)
        reference[1, 0] = 1.0
        obs["observation/eef_state"] = reference
        obs["actions"] = np.full((2, 14), 0.1, dtype=np.float32)
        inputs = g1d_policy.G1DInputs(16, 14, action_space="eef")(obs)
        np.testing.assert_allclose(inputs["actions"][:, 0], [0.1, 1.1])
        self.assertEqual(inputs["state"].shape, (16,))

    def test_lerobot_float_chw_matches_wire_uint8_hwc(self) -> None:
        obs = self.observation(16)
        obs["observation/images/left_wrist"] = np.ones((3, 12, 16), dtype=np.float32)
        inputs = g1d_policy.G1DInputs()(obs)
        self.assertEqual(inputs["image"]["left_wrist_0_rgb"].shape, (12, 16, 3))
        self.assertEqual(inputs["image"]["left_wrist_0_rgb"].dtype, np.uint8)
        np.testing.assert_array_equal(inputs["image"]["left_wrist_0_rgb"], 255)
        self.assertTrue(all(inputs["image_mask"].values()))
        self.assertNotIn("actions", inputs)

    def test_eef_actions_can_use_joint_feedback_state(self) -> None:
        obs = self.observation(16)
        obs["actions"] = np.zeros((5, 14), dtype=np.float32)
        inputs = g1d_policy.G1DInputs(state_dim=16, action_dim=14)(obs)
        self.assertEqual(inputs["state"].shape, (16,))
        self.assertEqual(inputs["actions"].shape, (5, 14))

    def test_bad_state_or_action_is_rejected(self) -> None:
        for key, value in (
            ("observation/state", np.zeros(14)),
            ("observation/state", np.full(16, np.inf)),
            ("actions", np.zeros((2, 14))),
            ("actions", np.zeros((0, 16))),
            ("actions", np.full((2, 16), np.nan)),
        ):
            with self.subTest(key=key, shape=value.shape), self.assertRaises(ValueError):
                obs = self.observation(16)
                obs[key] = value
                g1d_policy.G1DInputs()(obs)

    def test_bad_or_missing_images_are_rejected(self) -> None:
        for image in (np.zeros((12, 16)), np.full((12, 16, 3), 255.0), np.full((12, 16, 3), np.nan)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                obs = self.observation(16)
                obs["observation/images/head"] = image
                g1d_policy.G1DInputs()(obs)
        obs = self.observation(16)
        del obs["observation/images/right_wrist"]
        with self.assertRaises(KeyError):
            g1d_policy.G1DInputs()(obs)

    def test_short_or_nonfinite_output_is_rejected(self) -> None:
        for output in (np.zeros((2, 14)), np.full((2, 32), np.inf)):
            with self.subTest(shape=output.shape), self.assertRaises(ValueError):
                g1d_policy.G1DOutputs()(dict(actions=output))


if __name__ == "__main__":
    unittest.main()
