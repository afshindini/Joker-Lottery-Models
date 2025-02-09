"""Utility functions/classes for general purposes"""

from typing import List
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Dataset:
    """Dataset class to load/preprocess data"""

    path: str
    data: pd.DataFrame = field(init=False)
    headers: List[str] = field(init=False)
    length: int = field(init=False)

    def __post_init__(self) -> None:
        """Post initialization of the dataset class"""
        self.load_data()
        self.load_headers()
        self.length = len(self.data)

    def load_data(self) -> pd.DataFrame:
        """Load the dataset"""
        self.data = pd.read_excel(self.path)

    def load_headers(self) -> None:
        """Load the headers of the dataset"""
        self.headers = self.data.columns.tolist()
