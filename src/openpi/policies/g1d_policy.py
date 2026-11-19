"""G1-D transforms shared by training and inference.

Arm actions are already relative to the absolute observation anchoring the chunk.
Grippers are absolute values. Training per-row references are rebased to the
chunk-start state; the cloud adapter restores absolute joint targets after unnormalization.
"""

import dataclasses
from typing import Literal

import numpy as np


# Model slot names stay fixed; source keys describe G1-D cameras.
CAMERA_KEYS = {
    "base_0_rgb": "observation/images/high",
    "left_wrist_0_rgb": "observation/images/left_wrist",
    "right_wrist_0_rgb": "observation/images/right_wrist",
}


def _float_vector(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real array of shape {shape}, got {array.shape}")
    with np.errstate(over="ignore"):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} cannot be represented as float32")
    return array


def validate_state(value, state_dim: int) -> np.ndarray:
    """Check unbatched feedback and convert it to float32."""
    return _float_vector(value, (state_dim,), "State")


def validate_actions(
    value, action_dim: int, *, horizon: int | None = None, allow_padding: bool = False
) -> np.ndarray:
    """Check a real action chunk; optionally remove model-only padding."""
    actions = np.asarray(value)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("actions must have shape [H, D] with H > 0")
    if horizon is not None and actions.shape[0] != horizon:
        raise ValueError(f"Expected action horizon {horizon}, got {actions.shape[0]}")
    valid_dim = actions.shape[-1] >= action_dim if allow_padding else actions.shape[-1] == action_dim
    if not valid_dim:
        raise ValueError(f"Expected {'at least ' if allow_padding else ''}{action_dim} action dimensions")
    actions = actions[..., :action_dim]
    return _float_vector(actions, actions.shape, "actions")


def parse_rgb_image(value, *, wire_format: bool = False) -> np.ndarray:
    """Parse training images or enforce the online uint8 HWC contract."""
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError("Images must have shape HWC or CHW")
    if wire_format and (image.dtype != np.uint8 or image.shape[-1] != 3):
        raise ValueError("Online images must be RGB uint8 HWC")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError("Images must have three RGB channels and nonempty spatial dimensions")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Float images must contain finite values in [0, 1]")
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError("Images must be uint8 [0, 255] or floating point [0, 1]")
    return image


def _quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    scale = 0.5 if angle < 1e-8 else np.sin(angle / 2) / angle
    return np.concatenate(([np.cos(angle / 2)], rotvec * scale))


def _compose_rotation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, v1 = first[0], first[1:]
    w2, v2 = second[0], second[1:]
    quaternion = np.concatenate(([w1 * w2 - np.dot(v1, v2)], w1 * v2 + w2 * v1 + np.cross(v1, v2)))
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion = -quaternion
    length = np.linalg.norm(quaternion[1:])
    if length < 1e-8:
        return 2 * quaternion[1:]
    return quaternion[1:] * (2 * np.arctan2(length, quaternion[0]) / length)


def absolute_target(
    action: np.ndarray,
    reference: np.ndarray,
    action_space: Literal["joint", "eef"],
    *,
    eef_delta_frame: Literal["base", "tool"] = "base",
) -> np.ndarray:
    """Resolve against a fixed chunk reference, never the previous action."""
    if action_space not in ("joint", "eef") or eef_delta_frame not in ("base", "tool"):
        raise ValueError("Choose joint/eef action space and base/tool EEF delta frame")
    dim = 16 if action_space == "joint" else 14
    action = validate_state(action, dim).astype(np.float64)
    reference = validate_state(reference, dim).astype(np.float64)
    target = action.copy()
    if action_space == "joint":
        target[:7] += reference[:7]
        target[8:15] += reference[8:15]
    else:
        for start in (0, 7):
            anchor_rotation = _quaternion(reference[start + 3 : start + 6])
            delta_rotation = _quaternion(action[start + 3 : start + 6])
            displacement = action[start : start + 3]
            if eef_delta_frame == "tool":
                vector = anchor_rotation[1:]
                displacement = displacement + 2 * np.cross(
                    vector, np.cross(vector, displacement) + anchor_rotation[0] * displacement
                )
                rotation = _compose_rotation(anchor_rotation, delta_rotation)
            else:
                rotation = _compose_rotation(delta_rotation, anchor_rotation)
            target[start : start + 3] = reference[start : start + 3] + displacement
            target[start + 3 : start + 6] = rotation
    # Gripper channels are copied from action; never add the reference gripper.
    return validate_state(target, dim)


def rebase_actions(
    actions: np.ndarray,
    references: np.ndarray,
    action_space: Literal["joint", "eef"],
    *,
    eef_delta_frame: Literal["base", "tool"] = "base",
) -> np.ndarray:
    """Convert per-row state-relative actions to one chunk-start reference."""
    dim = 16 if action_space == "joint" else 14
    actions = validate_actions(actions, dim)
    references = validate_actions(references, dim, horizon=len(actions))
    anchor = references[0]
    if action_space == "joint":
        rebased = actions.astype(np.float64)
        rebased[:, :7] += references[:, :7].astype(np.float64) - anchor[:7]
        rebased[:, 8:15] += references[:, 8:15].astype(np.float64) - anchor[8:15]
        return validate_actions(rebased, dim)
    rebased = np.empty_like(actions)
    for index, (action, reference) in enumerate(zip(actions, references, strict=True)):
        target = absolute_target(action, reference, action_space, eef_delta_frame=eef_delta_frame)
        for start in (0, 7):
            anchor_q = _quaternion(anchor[start + 3 : start + 6].astype(np.float64))
            inverse_q = anchor_q.copy()
            inverse_q[1:] *= -1
            target_q = _quaternion(target[start + 3 : start + 6].astype(np.float64))
            displacement = target[start : start + 3] - anchor[start : start + 3]
            if eef_delta_frame == "tool":
                vector = inverse_q[1:]
                displacement = displacement + 2 * np.cross(
                    vector, np.cross(vector, displacement) + inverse_q[0] * displacement
                )
                rotation = _compose_rotation(inverse_q, target_q)
            else:
                rotation = _compose_rotation(target_q, inverse_q)
            target[start : start + 3] = displacement
            target[start + 3 : start + 6] = rotation
        rebased[index] = target
    return rebased


class GrippersLastToArmOrder:
    """Reorder offline joint data from [L7, R7, Lgrip, Rgrip] to the wire layout."""

    def __call__(self, data: dict) -> dict:
        order = [*range(7), 14, *range(7, 14), 15]
        return {**data, **{key: np.take(data[key], order, axis=-1) for key in ("observation/state", "actions")}}


@dataclasses.dataclass(frozen=True)
class G1DInputs:
    """Map canonical raw inputs to the model, retaining observation-relative actions.

    Joint layout: [left dq(7), left gripper, right dq(7), right gripper].
    EEF layout: [left dxyz(3), left drotvec(3), left gripper, right ...].
    Online state is a vector; training state can be a sequence for rebasing each
    row's relative action to the first observation. State itself stays absolute.
    """

    state_dim: int = 16
    action_dim: int = 16
    action_space: Literal["joint", "eef"] = "joint"
    eef_delta_frame: Literal["base", "tool"] = "base"

    absolute_joint_targets: bool = False

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        state_sequence = state if state.ndim == 2 else None
        if state_sequence is not None:
            if state_sequence.shape[0] == 0 or "actions" not in data:
                raise ValueError("A training state sequence requires a nonempty action chunk")
            state_sequence = validate_actions(state_sequence, self.state_dim)
            state = state_sequence[0]
        inputs = {
            "state": validate_state(state, self.state_dim),
            "image": {dest: parse_rgb_image(data[source]) for dest, source in CAMERA_KEYS.items()},
            "image_mask": {dest: np.True_ for dest in CAMERA_KEYS},
        }
        if "actions" in data:
            actions = validate_actions(data["actions"], self.action_dim)
            if self.absolute_joint_targets:
                actions = actions.copy()
                actions[:, :7] -= inputs["state"][:7]
                actions[:, 8:15] -= inputs["state"][8:15]
            elif state_sequence is not None:
                if self.action_space == "eef" and self.state_dim != 14:
                    references = data["observation/eef_state"]
                else:
                    references = state_sequence
                actions = rebase_actions(
                    actions, references, self.action_space, eef_delta_frame=self.eef_delta_frame
                )
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class G1DOutputs:
    """Remove padding; retain observation-relative actions and absolute grippers."""

    action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": validate_actions(data["actions"], self.action_dim, allow_padding=True)}
