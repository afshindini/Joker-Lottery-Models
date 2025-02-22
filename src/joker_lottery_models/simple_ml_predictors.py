"""Apply random forest classifier to predict the lottery numbers"""

from dataclasses import dataclass, field
from typing import List, Tuple, Any
from abc import ABC, abstractmethod

import logging
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from .utility import Dataset

logger = logging.getLogger(__name__)


@dataclass
class MLPredictor(ABC, Dataset):
    """This is base classifier for implementing ML models"""

    year: int = field(default=2025)
    week: int = field(default=1)
    day: int = field(default=1)

    def __post_init__(self) -> None:
        """Post initialization of the model class"""
        super().__post_init__()
        self.sanity_check()
        pd.options.mode.copy_on_write = True

    def sanity_check(self) -> None:
        """Check if the period of year/week/day are consistent"""
        if (
            self.year not in [2025, 2024, 2023, 2022]
            or self.week > 52
            or self.day < 1
            or self.day > 4
        ):
            logger.error("The period is not valid. Please check year/week/day values.")

    def data_selection(self, period: str = "year") -> pd.DataFrame:
        """Select data based on the period of data"""
        if period == "year":
            grouped_data = self.data.groupby("year").get_group(self.year)
        elif period == "week":
            grouped_data = self.data.groupby("week").get_group(self.week)
        elif period == "day":
            grouped_data = self.data.groupby("day").get_group(self.day)
        else:
            grouped_data = self.data
        grouped_data.loc[:, "whole_number"] = grouped_data.apply(
            lambda x: str(x["d1"])
            + str(x["d2"])
            + str(x["d3"])
            + str(x["d4"])
            + str(x["d5"])
            + str(x["d6"])
            + str(x["d7"]),
            axis=1,
        )
        return grouped_data

    @abstractmethod
    def prepare_data(self) -> Tuple[Any, Any]:
        """Prepare the data for the training purposes"""
        return None, None

    @abstractmethod
    def train_model(self) -> Any:
        """Train the model"""
        return None

    @abstractmethod
    def predict(self) -> Tuple[List[int], List[float]]:
        """Predict the lottery numbers using the trained model"""
        return [], []


@dataclass
class RandomForestPredictor(MLPredictor):
    """Implement Random Forest classifier for the lottery data"""

    model: Any = field(init=False)

    def prepare_data(self) -> Tuple[Any, Any]:
        """Prepare the data for the training purposes"""
        temp = self.data_selection("all")
        x_all = np.array(temp[self.headers[3:]].values[::-1][:-1])
        y_all = np.array(temp[self.headers[3:]].values[::-1][1:])
        return x_all, y_all

    def train_model(self) -> None:
        """Train the Random Forest classifier model"""
        x_all, y_all = self.prepare_data()
        self.model = RandomForestClassifier(n_estimators=1000)
        self.model.fit(x_all, y_all)

    def predict(self) -> Tuple[List[int], List[float]]:
        """Predict the lottery numbers using the trained model"""
        self.train_model()
        last_number = self.data_selection("all")[self.headers[3:]].values[::-1][-1]
        last_number_array = np.array([int(digit) for digit in last_number]).reshape(
            1, -1
        )
        logger.info(
            "Predicted numbers using Random Forest model: %s",
            self.model.predict(last_number_array)[0].tolist(),
        )
        return self.model.predict(last_number_array)[0].tolist(), [
            float(max(val[0])) for val in self.model.predict_proba(last_number_array)
        ]
