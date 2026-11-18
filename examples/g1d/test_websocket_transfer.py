"""Test G1-D WebSocket transfers as a client or start the corresponding server."""

import argparse
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy


def make_observation(contract: dict) -> dict:
    # Each arm contributes seven joint angles or six pose coordinates, then a gripper.
    joint = np.linspace(-0.3, 0.3, 16, dtype=np.float32)
    eef = np.linspace(0.1, 0.4, 14, dtype=np.float32)
    joint[[7, 15]] = eef[[6, 13]] = [0.25, 0.75]
    if contract["state_dim"] == joint.size:
        state, extra_key, extra = joint, "observation/eef_state", eef
    elif contract["state_dim"] == eef.size:
        state, extra_key, extra = eef, "observation/joint_state", joint
    else:
        raise ValueError(f"Unsupported G1-D test state dimension {contract['state_dim']}")

    rng = np.random.default_rng(42)
    images = {
        f"observation/images/{camera}": rng.integers(256, size=(224, 224, 3), dtype=np.uint8)
        for camera in ("head", "left_wrist", "right_wrist")
    }
    if set(images) != set(contract["image_keys"]):
        raise ValueError(f"Unexpected camera keys {contract['image_keys']}")
    return {**images, contract["state_key"]: state, extra_key: extra, "prompt": "Pick up the box."}


def robot_observation() -> dict:
    rng = np.random.default_rng(42)
    return {
        "state": np.linspace(0.1, 0.4, 12, dtype=np.float32),
        "images": {key: rng.integers(256, size=(224, 224, 3), dtype=np.uint8)
                   for key in ("head", "left_wrist", "right_wrist")},
        "prompt": "Pick up the box.",
        "_unitree": {"sequence": 1, "joint_positions": [0.1] * 14, "joint_velocities": [0.] * 14,
                     "waist_q3": [0.] * 3, "grip_q2": [0.2, 0.3], "last_action": {"status": "idle"}},
    }


def run(host: str, port: int) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    try:
        metadata = client.get_server_metadata()
        contract = metadata["g1d"]
        observation = robot_observation() if metadata.get("test_response") == "ones" else make_observation(contract)
        start = time.perf_counter()
        result = client.infer(observation)
        elapsed = time.perf_counter() - start
        actions = result["actions"]
        expected = (contract["action_horizon"], contract["action_dim"])
        if not isinstance(actions, np.ndarray) or actions.shape != expected:
            raise ValueError(f"Expected an action array of shape {expected}, got {np.shape(actions)}")
        if actions.dtype != np.float32 or not np.isfinite(actions).all():
            raise ValueError("Expected finite float32 actions")
        if metadata.get("test_response") == "ones":
            if result.get("observation_id") != observation["_unitree"]["sequence"]:
                raise ValueError("Response sequence mismatch")
            if not np.all(actions == 1):
                raise ValueError("Expected every action from the test server to equal 1")
            print("Verified all actions equal 1.")
        print(f"PASS: sent 3 RGB images and robot state; "
              f"received {contract['action_space']} actions {actions.shape} in {elapsed:.3f} s")
    finally:
        client._ws.close()  # WebsocketClientPolicy currently has no public close method.


def serve_ones(host: str, port: int) -> None:
    from openpi_client import base_policy

    from examples.g1d.real_env import G1DRealEnvironment
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    class OnesPolicy(base_policy.BasePolicy):
        def infer(self, obs: dict) -> dict:
            return {"actions": np.ones((10, 12), dtype=np.float32)}

    class RobotTestEnvironment(G1DRealEnvironment):
        def infer(self, obs: dict) -> dict:
            state = np.asarray(obs["state"])
            if state.shape != (12,) or state.dtype != np.float32 or not np.isfinite(state).all():
                raise ValueError("Expected finite float32 EEF state[12]")
            values = state.astype(np.float64)
            inputs = {"observation/state": ((values - values.mean()) / max(values.std(), 1e-6)).astype(np.float32),
                      "prompt": obs["prompt"]}
            for camera in ("head", "left_wrist", "right_wrist"):
                image = obs["images"][camera]
                if np.shape(image) != (224, 224, 3):
                    raise ValueError(f"Unexpected {camera} image shape")
                inputs[f"observation/images/{camera}"] = image
            return {**super().infer(inputs), "observation_id": obs["_unitree"]["sequence"]}

    env = RobotTestEnvironment(
        OnesPolicy(), action_space="eef", state_dim=12, action_dim=12,
        action_horizon=10, metadata={"test_response": "ones"},
    )
    env.metadata["g1d"].update(protocol="unitree-test-v1", state_key="state", image_container_key="images",
                               image_keys=["head", "left_wrist", "right_wrist"])
    WebsocketPolicyServer(policy=env, host=host, port=port, metadata=env.metadata).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Start the cloud-side server")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint", help="Serve predictions from this checkpoint")
    mode.add_argument("--no-checkpoint", action="store_true", help="Serve all-one test actions")
    parser.add_argument("--config", help="Training config for --checkpoint (default: pi05_g1d)")
    parser.add_argument("--host", help="Server bind address or client destination")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.serve != bool(args.checkpoint or args.no_checkpoint):
        parser.error("--serve requires either --checkpoint PATH or --no-checkpoint; omit both for client mode")
    if args.config and not args.checkpoint:
        parser.error("--config is only used with --checkpoint")
    if args.no_checkpoint:
        serve_ones(args.host or "0.0.0.0", args.port)
    elif args.checkpoint:
        from examples.g1d import main

        main.main(main.Args(config=args.config or "pi05_g1d", checkpoint=args.checkpoint,
                            host=args.host or "0.0.0.0", port=args.port))
    else:
        run(args.host or "127.0.0.1", args.port)
