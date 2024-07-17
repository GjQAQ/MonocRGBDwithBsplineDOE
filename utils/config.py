from pathlib import Path
import re
from typing import Sequence, Any

import yaml

__all__ = ['parse_config']


def _recursive_update(target: dict, source: dict) -> None:
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _recursive_update(target[k], v)
        else:
            target[k] = v


def _rectify_float(d):
    if isinstance(d, dict):
        for k in d:
            d[k] = _rectify_float(d[k])
        return d
    elif isinstance(d, list):
        return [_rectify_float(item) for item in d]
    elif isinstance(d, str):
        try:
            return float(d)
        except ValueError:
            return d
    else:
        return d


def parse_config(files: Sequence[str | Path], override: bool = False) -> dict[str, Any]:
    final_config = {}
    for file_path in files:
        with open(file_path, 'r') as f:
            config_item = yaml.load(f, yaml.FullLoader)
        if not isinstance(config_item, dict):
            raise RuntimeError(f'Wrong configuration format: must be a YAML object')

        if not override:
            for k in config_item:
                if k in final_config:
                    raise RuntimeError(f'Duplicate config key: {k}')

        _rectify_float(config_item)
        _recursive_update(final_config, config_item)
    return final_config
