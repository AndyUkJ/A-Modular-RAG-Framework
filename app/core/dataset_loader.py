import json
from pathlib import Path
from typing import Any, Dict, List


class DatasetLoader:
    """可扩展的数据集加载器基类"""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def load(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class HotpotQALoader(DatasetLoader):
    """加载 HotpotQA 数据集 (JSON Lines)"""

    def load(self) -> List[Dict[str, Any]]:
        path = Path(self.cfg["path"])
        index = self.cfg.get("index", 0)
        count = self.cfg.get("count", 1)

        if not path.exists():
            raise FileNotFoundError(f"HotpotQA dataset not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                # 大 JSON 数组
                data = json.load(f)
            else:
                # JSON Lines
                data = [json.loads(line) for line in f]

        print(f"[DatasetLoader] loaded {len(data)} samples, index={index}, count={count}")
        
        # ✅ 截取 index/count 范围
        if count == -1:
            return data[index:]
        else:
            return data[index:index + count]


# 注册表：可以扩展更多数据源
DATASET_REGISTRY = {
    "hotpotqa": HotpotQALoader,
    # "musique": MusiQueLoader,
    # "wikipedia": WikipediaLoader,
    # "api": APILoader,
}


def build_dataset_loader(cfg: Dict[str, Any]) -> DatasetLoader:
    ds_type = cfg.get("type")
    if ds_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: {ds_type}")
    return DATASET_REGISTRY[ds_type](cfg)
