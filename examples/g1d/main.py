"""Start G1-D cloud inference; the training config owns the data contract."""

import argparse
import dataclasses
import logging

from examples.g1d import real_env
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Args:
    config: str
    checkpoint: str
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str | None = None
    pytorch_device: str | None = None


def main(args: Args) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    env = real_env.create_environment(
        args.config,
        args.checkpoint,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    logging.info("G1-D server listening on %s:%s", args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=env, host=args.host, port=args.port, metadata=env.metadata
    ).serve_forever()


def parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt")
    parser.add_argument("--pytorch-device")
    return Args(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(parse_args())
