"""Read local LeRobotDataset v3 shards while keeping the existing v2 dependency."""

from bisect import bisect_right
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq


class LocalLeRobotV3Dataset:
    def __init__(self, root: Path, info: dict, sequence_keys, horizon: int, prompt_from_task: bool):
        self.root = root
        self.info = info
        self.sequence_keys = set(sequence_keys)
        self.horizon = horizon
        self.prompt_from_task = prompt_from_task
        features = info["features"]
        unknown = self.sequence_keys.difference(features)
        if unknown or horizon < 1:
            raise ValueError(f"Invalid action window: horizon={horizon}, unknown features={unknown}")
        if self.sequence_keys.intersection(k for k, f in features.items() if f["dtype"] in ("image", "video")):
            raise ValueError("Only low-dimensional action sequence keys are supported")
        self.video_keys = [k for k, f in features.items() if f["dtype"] == "video"]
        columns = ["dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index"]
        for key in self.video_keys:
            columns.extend(f"videos/{key}/{field}" for field in ("chunk_index", "file_index", "from_timestamp"))
        self.episodes = sorted(
            (row for path in (root / "meta/episodes").glob("chunk-*/*.parquet")
             for row in pq.read_table(path, columns=columns).to_pylist()), key=lambda row: row["dataset_from_index"]
        )
        self.ends = [ep["dataset_to_index"] for ep in self.episodes]
        if not self.ends or self.ends[-1] != info["total_frames"]:
            raise ValueError("Episode index does not cover total_frames")
        tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
        self.tasks = dict(zip(tasks["task_index"], tasks.index, strict=True))
        self._table = None
        self._path = None

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        from lerobot.common.datasets.video_utils import decode_video_frames

        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = self.episodes[bisect_right(self.ends, index)]
        path = self.root / self.info["data_path"].format(chunk_index=ep["data/chunk_index"],
                                                         file_index=ep["data/file_index"])
        if path != self._path:
            self._table = pq.read_table(path)
            self._path = path
        local = index - self._table["index"][0].as_py()
        size = min(self.horizon, ep["dataset_to_index"] - index)
        sample = {}
        for key, feature in self.info["features"].items():
            if key in self.video_keys:
                continue
            col = self._table[key]
            if key in self.sequence_keys:
                values = col.slice(local, size).to_pylist()
                sample[key] = np.asarray(values + [values[-1]] * (self.horizon - size), dtype=feature["dtype"])
                sample[key + "_is_pad"] = np.arange(self.horizon) >= size
            elif feature["dtype"] == "image":
                item = col[local].as_py()
                source = io.BytesIO(item["bytes"]) if item["bytes"] is not None else self.root / item["path"]
                with Image.open(source) as image:
                    sample[key] = np.asarray(image.convert("RGB")).transpose(2, 0, 1) / np.float32(255)
            else:
                sample[key] = np.asarray(col[local].as_py(), dtype=feature["dtype"])
        for key in self.video_keys:
            prefix = f"videos/{key}/"
            video = self.root / self.info["video_path"].format(video_key=key, chunk_index=ep[prefix + "chunk_index"],
                                                               file_index=ep[prefix + "file_index"])
            timestamp = ep[prefix + "from_timestamp"] + float(sample["timestamp"])
            sample[key] = decode_video_frames(video, [timestamp], tolerance_s=1e-4, backend="pyav")[0]
        sample["task"] = self.tasks[int(sample["task_index"])]
        if self.prompt_from_task:
            sample["prompt"] = sample["task"]
        return sample
