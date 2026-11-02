"""Native deployment compatibility exports; SIMPLE wire implementation lives in rhsci."""

import json
import torch
import numpy as np

from base64 import b64decode, b64encode
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from fastapi.responses import JSONResponse
from dataclasses import dataclass
from typing import Union, Dict, Any, List
from pathlib import Path

from rhsci.inference.protocol import (
    Message,
    RequestMessage,
    ResponseMessage,
    convert_numpy_in_dict,
    numpy_deserialize,
    numpy_serialize,
)
