"""Inference-only entrypoint and stock WebSocket client compatibility."""

import asyncio
import contextlib
import io
import unittest
from unittest import mock

import numpy as np
from openpi_client import websocket_client_policy
import websockets.asyncio.server

from examples.g1d import main
from examples.g1d.test_real_env import RecordingPolicy
from examples.g1d.test_real_env import environment
from examples.g1d.test_real_env import observation
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class MainTest(unittest.TestCase):
    def test_training_settings_are_not_cli_options(self) -> None:
        for flag, value in (
            ("--action-space", "eef"), ("--state-dim", "16"),
            ("--action-horizon", "2"), ("--eef-delta-frame", "base"),
        ):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main.parse_args(["--config", "test", "--checkpoint", "/checkpoint", flag, value])
                self.assertEqual(error.exception.code, 2)

    def test_invalid_port_is_rejected_before_loading_model(self) -> None:
        with mock.patch.object(main.real_env, "create_environment") as load:
            with self.assertRaises(ValueError):
                main.main(main.Args(config="test", checkpoint="/checkpoint", port=0))
        load.assert_not_called()

    def test_main_only_loads_environment_and_starts_transport(self) -> None:
        args = main.parse_args([
            "--config", "my_g1d", "--checkpoint", "/checkpoint", "--port", "9000",
            "--default-prompt", "task", "--pytorch-device", "cuda:0",
        ])
        env = environment(RecordingPolicy())
        with (
            mock.patch.object(main.real_env, "create_environment", return_value=env) as load,
            mock.patch.object(main.websocket_policy_server, "WebsocketPolicyServer") as server,
        ):
            main.main(args)
        load.assert_called_once_with("my_g1d", "/checkpoint", default_prompt="task", pytorch_device="cuda:0", rtc=False)
        server.assert_called_once_with(policy=env, host="0.0.0.0", port=9000, metadata=env.metadata)
        server.return_value.serve_forever.assert_called_once()


class WebsocketTest(unittest.IsolatedAsyncioTestCase):
    async def test_stock_client_receives_metadata_and_absolute_joint_chunks(self) -> None:
        env = environment(RecordingPolicy())
        handler = WebsocketPolicyServer(env, metadata=env.metadata)
        async with websockets.asyncio.server.serve(handler._handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            def run_client() -> None:
                client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)
                try:
                    metadata = client.get_server_metadata()
                    self.assertEqual(metadata["g1d"]["action_horizon"], 2)
                    for state_value in (10, 20):
                        obs = observation()
                        obs["observation/state"][:] = state_value
                        result = client.infer(obs)
                        self.assertEqual(result["actions"].shape, (2, 16))
                        np.testing.assert_allclose(result["actions"][:, :7], state_value + 0.1)
                        np.testing.assert_allclose(result["actions"][:, 8:15], state_value + 0.1)
                        np.testing.assert_allclose(result["actions"][:, [7, 15]], 0.1)
                        self.assertIn("infer_ms", result["server_timing"])
                finally:
                    client._ws.close()
            await asyncio.wait_for(asyncio.to_thread(run_client), timeout=10)


if __name__ == "__main__":
    unittest.main()
