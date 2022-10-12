from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class Metadata:
    name: str
    path: Path
    instances: int
    features: int
    nominal_features: List[str]
    classes: int
    IR: float
