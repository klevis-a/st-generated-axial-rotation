"""This module contains various utilities for interacting with JSON."""

import json
from pathlib import Path
from collections import namedtuple
from typing import Dict, Any, Tuple, Union


def _json_object_hook(d: Dict[Any, Any]) -> Tuple:
    """Transform a dictionary into a namedtuple."""
    return namedtuple('params', d.keys())(*d.values())


def json2obj(file: Union[str, Path]) -> Tuple:
    """Read a json file and create a namedtuple from its entries."""
    with open(file) as f:
        return json.load(f, object_hook=_json_object_hook)


def get_params(params_file):
    """Return a namedtuple object with the parameters contained in params_file."""
    return  json2obj(params_file)
