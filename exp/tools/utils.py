import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def str2bool(val):
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise RuntimeError(f"Expected a boolean value, got: {val}")


def str2dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_sweep_lengths(step: int, max_L: int) -> List[int]:
    if step <= 0:
        raise ValueError("sweep_step must be > 0")
    if max_L <= 1:
        raise ValueError("max_L must be > 1")
    Ls = list(range(step, max_L + 1, step))
    if Ls[-1] != max_L:
        Ls.append(max_L)
    return sorted(list(dict.fromkeys(Ls)))


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(v)


def resolve_split(ds_container: DatasetDict, split_expr: str):
    if "[" not in split_expr:
        return ds_container[split_expr]

    base, bracket = split_expr.split("[", 1)
    bracket = bracket.rstrip("]")
    if base not in ds_container:
        raise ValueError(f"Split base '{base}' not found in local dataset.")
    ds = ds_container[base]
    total = len(ds)

    def parse_bound(tok: str):
        tok = tok.strip()
        if tok == "":
            return None
        if tok.endswith("%"):
            return int(float(tok[:-1]) / 100.0 * total)
        return int(tok)

    start_s, end_s = bracket.split(":", 1)
    start = parse_bound(start_s)
    end = parse_bound(end_s)
    if start is None:
        start = 0
    if end is None or end > total:
        end = total
    start = max(0, start)
    end = max(start, min(end, total))
    return ds.select(range(start, end))


def load_ds(dataset_path: Optional[str], split: str) -> Dataset:
    if not dataset_path:
        raise ValueError("dataset_path is required")

    p = Path(dataset_path).expanduser()

    if p.exists() and p.is_dir():
        parquet_files = list(p.glob("*.parquet"))
        if parquet_files:
            data_files = {}
            if list(p.glob("train-*.parquet")):
                data_files["train"] = str(p / "train-*.parquet")
            if list(p.glob("validation-*.parquet")):
                data_files["validation"] = str(p / "validation-*.parquet")
            if list(p.glob("test-*.parquet")):
                data_files["test"] = str(p / "test-*.parquet")

            if not data_files:
                raise ValueError(f"Found parquet files in {p} but none match train/validation/test patterns.")

            ds_container = load_dataset("parquet", data_files=data_files)
            return resolve_split(ds_container, split)
    
        try:
            loaded = load_from_disk(p)
            if isinstance(loaded, Dataset):
                ds_container = DatasetDict({"train": loaded})
            elif isinstance(loaded, DatasetDict):
                ds_container = loaded
            else:
                raise ValueError(f"Unsupported dataset object from disk: {type(loaded)}")
            return resolve_split(ds_container, split)
        except Exception as e:
            print(f"!! Warning: failed to load dataset from '{p}': {e}")
            print("!! Fallback to HuggingFace load_dataset(...)")
    else:
        print(f"!! Warning: dataset_path '{p}' does not exist.")
        print("!! Fallback to HuggingFace load_dataset(...)")

    return load_dataset(dataset_path, split=split)


def to_jsonable(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x  